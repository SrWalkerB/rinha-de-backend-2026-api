// Package knn holds the reference vectors and runs a k-nearest-neighbors search
// to score transactions. Vectors are quantized to uint16 so the full 3M-row
// dataset fits in the per-instance memory budget (~84MB) while keeping enough
// precision that exact search matches the float64 ground truth (uint8's 255
// buckets flipped ~0.4% of near-boundary cases; see docs/performance/07).
//
// Three search strategies share the same quantized storage and the same K-NN
// result contract (Score). They are selected via Build:
//   - brute: scan every row (exact; the original, used for small N / fallback).
//   - vptree: metric-tree exact search (same neighbors as brute, fewer visits).
//   - ivf: inverted-file approximate search (k-means cells; fastest, ~recall<1).
package knn

import (
	"math"

	"rinha-fraud/internal/vectorize"
)

// K is the number of nearest neighbors used to score a transaction.
const K = 5

// Threshold: a transaction is approved when fraud_score < Threshold.
const Threshold = 0.6

// buildThreshold: below this many rows, Score always brute-forces regardless of
// the configured mode. Keeps small indices (unit tests use 5..100 rows) exact
// and avoids building a tree/clusters over a handful of points.
const buildThreshold = 2048

// quantize maps a normalized dimension to a uint16 bucket on the data's native
// grid. The reference vectors are exactly 4-decimal, so round(v*10000) stores
// them losslessly; dequant then reproduces the parsed float64 bit-for-bit.
//
//	-1 (sentinel, no last_transaction) -> 0
//	[0, 1]                             -> [1, 10001]
//
// Distance is computed in float64 (see dist2) against the un-quantized query,
// so an exact (brute) search reproduces the float64 ground-truth 5-NN with no
// quantization flips. The ~144 flips under the old ×65534 scale came from
// quantizing the full-precision query — not the reference buckets (docs 07/08).
func quantize(v float64) uint16 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		v = 1
	}
	return uint16(math.Round(v*10000)) + 1
}

// maxBucket is the largest stored bucket: quantize(1) = round(1*10000)+1.
const maxBucket = 10001

// dequantTab maps a stored bucket to its exact float64 value, precomputed once
// so the hot distance loop does a table lookup instead of a per-element division
// (the /10000 division dominated p99 — see docs/performance/08). Bucket 0 is the
// sentinel -1; buckets [1,10001] map to k/10000 (the exact ground-truth value,
// bit-identical to what dividing would produce).
var dequantTab = func() [maxBucket + 1]float64 {
	var t [maxBucket + 1]float64
	t[0] = -1
	for u := 1; u <= maxBucket; u++ {
		t[u] = float64(u-1) / 10000
	}
	return t
}()

// dequant reverses quantize via the precomputed table.
func dequant(u uint16) float64 {
	return dequantTab[u]
}

type searchMode uint8

const (
	modeBrute searchMode = iota
	modeVPTree
	modeIVF
)

// BuildConfig selects and parameterizes the search structure built by Build.
type BuildConfig struct {
	Mode   string // "brute" | "vptree" | "ivf" (anything else => brute)
	NList  int    // IVF: number of k-means cells
	NProbe int    // IVF: cells visited per query
	Iters  int    // IVF: k-means iterations
}

// Index holds quantized reference vectors (flat, row-major) and a fraud bitset.
// After Add-ing all rows, call Build to construct an accelerator; Score then
// dispatches to it. Until Build is called (or for tiny N) Score is brute-force.
type Index struct {
	data []uint16 // n * vectorize.Dims quantized values
	bits []uint64 // fraud bitset, one bit per row (1 = fraud)
	n    int

	mode searchMode
	vp   *vpTree
	ivf  *ivfIndex
}

// NewIndex pre-allocates storage for capacityHint rows.
func NewIndex(capacityHint int) *Index {
	if capacityHint < 0 {
		capacityHint = 0
	}
	return &Index{
		data: make([]uint16, 0, capacityHint*vectorize.Dims),
		bits: make([]uint64, 0, (capacityHint+63)/64),
	}
}

// Add quantizes vec and appends it with its fraud label.
func (ix *Index) Add(vec [vectorize.Dims]float64, fraud bool) {
	for i := 0; i < vectorize.Dims; i++ {
		ix.data = append(ix.data, quantize(vec[i]))
	}
	word := ix.n >> 6
	for len(ix.bits) <= word {
		ix.bits = append(ix.bits, 0)
	}
	if fraud {
		ix.bits[word] |= 1 << uint(ix.n&63)
	}
	ix.n++
}

// Len reports how many reference vectors are stored.
func (ix *Index) Len() int { return ix.n }

func (ix *Index) isFraud(i int) bool {
	return ix.bits[i>>6]&(1<<uint(i&63)) != 0
}

// Build constructs the search accelerator selected by cfg. Must be called after
// all rows are Add-ed and before concurrent Score calls. For n < buildThreshold
// it forces brute-force (keeps small indices exact). Safe to call once.
func (ix *Index) Build(cfg BuildConfig) {
	if ix.n < buildThreshold {
		ix.mode = modeBrute
		return
	}
	switch cfg.Mode {
	case "vptree":
		ix.vp = buildVPTree(ix)
		ix.mode = modeVPTree
	case "ivf":
		ix.ivf = buildIVF(ix, cfg)
		ix.mode = modeIVF
	default: // "brute", "" or unknown
		ix.mode = modeBrute
	}
}

// SetNProbe overrides how many IVF cells each query scans. No-op unless the
// index is IVF and np > 0. Clamped to [1, nlist] and maxProbe. Lets a baked
// (pre-built) index be retuned at startup without rebuilding.
func (ix *Index) SetNProbe(np int) {
	if ix.ivf == nil || np <= 0 {
		return
	}
	if np > ix.ivf.nlist {
		np = ix.ivf.nlist
	}
	if np > maxProbe {
		np = maxProbe
	}
	ix.ivf.nprobe = np
}

// dist2 returns the squared euclidean distance between the un-quantized query q
// and stored row `row`. The stored uint16 ref is dequantized to its exact
// 4-decimal float64, so the result matches the float64 ground-truth distance.
// Squared distance preserves nearest-neighbor ordering, so no sqrt is needed for
// ranking; the VP-tree converts to true distance only where the triangle
// inequality requires it.
func (ix *Index) dist2(q *[vectorize.Dims]float64, row int) float64 {
	off := row * vectorize.Dims
	data := ix.data
	var dist float64
	for d := 0; d < vectorize.Dims; d++ {
		diff := q[d] - dequant(data[off+d])
		dist += diff * diff
	}
	return dist
}

// rowDist2 is the squared distance between two stored rows (used while building
// the VP-tree). Both rows are dequantized to float64.
func (ix *Index) rowDist2(a, b int) float64 {
	oa, ob := a*vectorize.Dims, b*vectorize.Dims
	data := ix.data
	var dist float64
	for d := 0; d < vectorize.Dims; d++ {
		diff := dequant(data[oa+d]) - dequant(data[ob+d])
		dist += diff * diff
	}
	return dist
}

// topK is a tiny sorted buffer (ascending distance) of the K nearest seen so far.
// Shared by every search strategy so they all produce the identical fraud score
// given the same set of visited rows.
type topK struct {
	dist  [K]float64
	fraud [K]bool
}

func newTopK() topK {
	var t topK
	for i := range t.dist {
		t.dist[i] = math.MaxFloat64
	}
	return t
}

// worst is the distance of the current K-th nearest (the pruning bound).
func (t *topK) worst() float64 { return t.dist[K-1] }

// consider inserts (dist, fraud) into the buffer if it beats the current K-th.
func (t *topK) consider(dist float64, fraud bool) {
	if dist >= t.dist[K-1] {
		return
	}
	pos := K - 1
	for pos > 0 && t.dist[pos-1] > dist {
		t.dist[pos] = t.dist[pos-1]
		t.fraud[pos] = t.fraud[pos-1]
		pos--
	}
	t.dist[pos] = dist
	t.fraud[pos] = fraud
}

func (t *topK) fraudScore() float64 {
	fraudCount := 0
	for i := 0; i < K; i++ {
		if t.fraud[i] {
			fraudCount++
		}
	}
	return float64(fraudCount) / float64(K)
}

// Score returns the fraud fraction among the K nearest reference vectors. The
// query is NOT quantized — it stays full-precision float64 and is compared
// against the dequantized refs (exact reproduction of the ground-truth 5-NN).
func (ix *Index) Score(query [vectorize.Dims]float64) float64 {
	if ix.n == 0 {
		return 0
	}

	switch ix.mode {
	case modeVPTree:
		return ix.vp.search(ix, &query)
	case modeIVF:
		return ix.ivf.search(ix, &query)
	default:
		return ix.bruteScore(&query)
	}
}

// bruteScore scans every row — exact, O(n). The fallback and the small-N path.
func (ix *Index) bruteScore(q *[vectorize.Dims]float64) float64 {
	tk := ix.bruteTopK(q)
	return tk.fraudScore()
}

// bruteTopK is the exact K nearest by full scan (used by bruteScore and by the
// VP-tree/IVF correctness tests as the source of truth).
func (ix *Index) bruteTopK(q *[vectorize.Dims]float64) topK {
	tk := newTopK()
	for idx := 0; idx < ix.n; idx++ {
		tk.consider(ix.dist2(q, idx), ix.isFraud(idx))
	}
	return tk
}
