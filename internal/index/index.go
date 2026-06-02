// Package index holds the 3M labeled reference vectors and answers a 5-nearest-
// neighbors fraud query in microseconds, fast enough to sustain the load test.
//
// Why this shape. A full float64 brute scan reproduces the official ground truth
// (0 failures) but costs ~870ms p99. Exact spatial pruning does NOT help here:
// the data is "spread" (the 5th neighbor sits ~0.5 away in ~10 effective dims),
// so a k-d tree still visits a large fraction of the data, and — worse — its
// reordered layout makes every visit a cache miss, which collapses under
// concurrent load. The decision, though, is only a majority vote over 5
// neighbors, which is robust to small perturbations of the neighbor set. So we
// use an approximate index that scans FEW rows CONTIGUOUSLY:
//
//   - 16 hard buckets keyed by the four ≥1.0-gap dims (is_online, card_present,
//     unknown_merchant, last_transaction-null). Each bucket is a contiguous range.
//   - Within each bucket, k-means partitions the rows into `nlist` cells; rows are
//     stored contiguously by cell (CSR offsets). A query computes its distance to
//     the bucket's centroids, scans the `nprobe` nearest cells (a handful of
//     contiguous, prefetch-friendly rows), and — only when the running 5th
//     distance could still be beaten across a bucket boundary — probes the
//     relevant neighbouring buckets.
//
// Storage is uint16 on the data's native 4-decimal grid; distance is float64 with
// the query un-quantized, so each scanned candidate's distance is exact. nprobe is
// a runtime knob (recall vs latency) retunable without a rebuild.
package index

import (
	"math"

	"rinha-fraud/internal/vectorize"
)

// Dims is the vector dimensionality (kept identical to the vectorizer).
const Dims = vectorize.Dims

// K is the number of nearest neighbors used to score a transaction.
const K = 5

// Threshold: a transaction is approved when fraud_score < Threshold.
const Threshold = 0.6

// numBuckets = 2^4: is_online, card_present, unknown_merchant, null flag.
const numBuckets = 16

// maxProbe caps nprobe (low AND high tier) so the per-query probe buffer lives
// on the stack (zero-allocation Score). nprobe and nprobeHigh are clamped to this.
const maxProbe = 512

// simdTailPad is the number of zero uint16 appended after the SoA data array so
// the int16 SIMD kernel's last 8-row m128 load can never read past the slice.
const simdTailPad = 8

// simdChunk is the per-call row batch for the SIMD cell scan (multiple of 8); the
// distance buffer is a [simdChunk]int32 on the stack (zero-allocation).
const simdChunk = 512

// candK is how many candidate rows (smallest integer distance) the scan collects
// before the exact float64 refine. Must be >> K so the true float 5-NN is always
// within the integer-nearest candK (the int distance differs from the float only
// by the tiny query-grid rounding). 64 keeps the refine cheap while leaving a huge
// safety margin over K=5; the diag E-gate confirms detection is unchanged.
const candK = 64

// --- uint16 quantization (native 4-decimal grid) -------------------------

const maxBucketVal = 10001 // quantize(1) = round(1*10000)+1

// quantize maps a normalized dimension to a uint16 on the data's native grid.
//
//	-1 (sentinel, no last_transaction) -> 0
//	[0, 1]                             -> [1, 10001]
func quantize(v float64) uint16 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		v = 1
	}
	return uint16(math.Round(v*10000)) + 1
}

// dequantTab maps a stored bucket to its exact float64 value, precomputed once so
// the hot distance loop does a table lookup instead of a per-element division.
var dequantTab = func() [maxBucketVal + 1]float64 {
	var t [maxBucketVal + 1]float64
	t[0] = -1
	for u := 1; u <= maxBucketVal; u++ {
		t[u] = float64(u-1) / 10000
	}
	return t
}()

func dequant(u uint16) float64 { return dequantTab[u] }

// bucketOf returns the hard-bucket index for a vector: the four dims where any
// mismatch costs ≥1.0 in squared distance.
func bucketOf(v *[Dims]float64) int {
	b := 0
	if v[9] >= 0.5 { // is_online
		b |= 8
	}
	if v[10] >= 0.5 { // card_present
		b |= 4
	}
	if v[11] >= 0.5 { // unknown_merchant
		b |= 2
	}
	if v[5] < 0 { // last_transaction null (dims 5,6 are −1 together)
		b |= 1
	}
	return b
}

// bucketPenalty is a lower bound on the squared distance from q to ANY row in
// hard bucket `target`: the forced contribution of the four key dims. Used to
// skip whole buckets cheaply during the cross-bucket pass.
func bucketPenalty(q *[Dims]float64, target int) float64 {
	var p float64
	on := 0
	if q[9] >= 0.5 {
		on = 1
	}
	if on != (target>>3)&1 {
		p += 1
	}
	cp := 0
	if q[10] >= 0.5 {
		cp = 1
	}
	if cp != (target>>2)&1 {
		p += 1
	}
	um := 0
	if q[11] >= 0.5 {
		um = 1
	}
	if um != (target>>1)&1 {
		p += 1
	}
	qnull := 0
	if q[5] < 0 {
		qnull = 1
	}
	if tnull := target & 1; qnull != tnull {
		if qnull == 1 {
			p += 2
		} else {
			p += (q[5]+1)*(q[5]+1) + (q[6]+1)*(q[6]+1)
		}
	}
	return p
}

// --- the index -----------------------------------------------------------

// Index holds the reference vectors reordered into (bucket, cell) order, a fraud
// bitset, the per-bucket k-means centroids, and CSR cell offsets. Build it with a
// Builder (or LoadIndex); Score is then safe for concurrent callers.
type Index struct {
	data      []uint16 // n*Dims quantized, reordered by (bucket, cell)
	fraud     []uint64 // fraud bitset over reordered rows
	n         int
	nlist     int       // k-means cells per bucket
	centroids []float64 // numBuckets*nlist*Dims, in the dequantized value space
	// centroidsI16 is the quantized SoA (dim-major per bucket) twin of centroids,
	// derived (never serialized): centroidsI16[b*Dims*nlist + d*nlist + c] =
	// quantize(centroids[(b*nlist+c)*Dims+d]). The int16 SIMD kernel (distSoAi16AVX2)
	// scans the nlist centroids 8 at a time to pick the nprobe cells — replacing the
	// scalar O(nlist) float64 centroid scan that dominated per-query CPU (the p99
	// driver). Cell SELECTION only needs ordering, so the integer (query-quantized)
	// distance is enough: exactness still comes from the float64 refine of candidates.
	// +simdTailPad zero tail so the kernel's last 8-cell load can't read past the slice.
	centroidsI16 []uint16
	cellStart    []int32 // CSR: rows of global cell i = [cellStart[i], cellStart[i+1]); len numBuckets*nlist+1

	bucketStart [17]int32 // rows of bucket b = [bucketStart[b], bucketStart[b+1])

	nprobe  int // cells scanned per bucket per query (cheap tier, runtime knob)
	maxScan int // per-query row cap (0 = unlimited); tail guardrail

	// Adaptive (two-tier) nprobe. The cheap pass runs at nprobe; if its 5th-NN
	// search radius reaches triggerRadius the query is re-run at nprobeHigh (the
	// only queries that miss are those whose true 5-NN crosses a hard-bucket
	// boundary, i.e. radius near/above 1.0). nprobeHigh<=nprobe disables the high
	// pass (zero-value default => behaviour identical to the single-tier search).
	nprobeHigh        int        // cells/bucket for the escalated pass (0/<=nprobe = off)
	nprobeHighByCount [K + 1]int // optional per-cheap-vote high nprobe (0 = use nprobeHigh)
	triggerRadius     float64    // escalate when topK.worst() >= this (squared distance)
	triggerMargin     float64    // optional AND-gate: also require |count-K*Threshold|<=margin (<0 = off)
}

// SearchTrace describes the two-tier search decision. It is diagnostic-only and
// is intentionally not used by the serving hot path.
type SearchTrace struct {
	Escalated       bool
	CheapFraudCount int
	CheapScanned    int
	HighScanned     int
}

// Len reports how many reference vectors are stored.
func (ix *Index) Len() int { return ix.n }

// NList reports the number of k-means cells per bucket.
func (ix *Index) NList() int { return ix.nlist }

// SetNProbe sets how many cells are scanned per probed bucket (clamped to
// [1, min(nlist, maxProbe)]). ≤0 leaves it unchanged.
func (ix *Index) SetNProbe(np int) {
	if np <= 0 {
		return
	}
	if np > ix.nlist {
		np = ix.nlist
	}
	if np > maxProbe {
		np = maxProbe
	}
	ix.nprobe = np
}

// SetMaxScan sets the per-query row-scan cap (≤0 means unlimited).
func (ix *Index) SetMaxScan(n int) {
	if n < 0 {
		n = 0
	}
	ix.maxScan = n
}

// SetNProbeHigh sets the escalated-pass nprobe (clamped to [1, min(nlist,
// maxProbe)]). A value ≤ the cheap nprobe disables escalation (the search stays
// single-tier and bit-identical to before).
func (ix *Index) SetNProbeHigh(np int) {
	if np <= 0 {
		ix.nprobeHigh = 0
		for i := range ix.nprobeHighByCount {
			ix.nprobeHighByCount[i] = 0
		}
		return
	}
	if np > ix.nlist {
		np = ix.nlist
	}
	if np > maxProbe {
		np = maxProbe
	}
	ix.nprobeHigh = np
	for i := range ix.nprobeHighByCount {
		ix.nprobeHighByCount[i] = np
	}
}

// SetNProbeHighForCount overrides the escalated-pass nprobe for a specific
// cheap-pass fraud count. This lets boundary classes spend different amounts of
// CPU while preserving the old single HIGH knob as the default.
func (ix *Index) SetNProbeHighForCount(count, np int) {
	if count < 0 || count > K {
		return
	}
	if np <= 0 {
		ix.nprobeHighByCount[count] = 0
		return
	}
	if np > ix.nlist {
		np = ix.nlist
	}
	if np > maxProbe {
		np = maxProbe
	}
	ix.nprobeHighByCount[count] = np
}

// SetTriggerRadius sets the squared-distance threshold on the cheap pass's 5th-NN
// (topK.worst()) above which the query escalates to nprobeHigh. ≤0 leaves it
// unchanged. Cross-bucket misses sit just above 1.0, so values near 1.0 fire
// rarely; lower values fire more (and risk moving p99).
func (ix *Index) SetTriggerRadius(r float64) {
	if r <= 0 {
		return
	}
	ix.triggerRadius = r
}

// SetTriggerMargin sets an optional AND-gate: escalate only when the cheap vote
// is also near the 0.6 threshold (|fraudCount - K*Threshold| <= margin). A
// negative margin disables the gate (radius alone decides).
func (ix *Index) SetTriggerMargin(m float64) {
	ix.triggerMargin = m
}

func (ix *Index) isFraud(i int) bool {
	return ix.fraud[i>>6]&(1<<uint(i&63)) != 0
}

// bucketOfRow returns the hard bucket that contains global row `row`. Used by the
// generic dist2 (diagnostics); the hot path passes the bucket context directly.
func (ix *Index) bucketOfRow(row int) int {
	for b := 0; b < numBuckets; b++ {
		if row < int(ix.bucketStart[b+1]) {
			return b
		}
	}
	return numBuckets - 1
}

// dist2 returns the exact squared euclidean distance between the un-quantized query
// q and stored row `row`. Data is SoA (dim-major) per bucket, so this resolves the
// row's bucket then reads the strided dim values. Used by diagnostics (BruteDebug);
// the hot path uses dist2SoA with the bucket context already in hand.
func (ix *Index) dist2(q *[Dims]float64, row int) float64 {
	b := ix.bucketOfRow(row)
	bLo := int(ix.bucketStart[b])
	bRows := int(ix.bucketStart[b+1]) - bLo
	if bRows == 0 {
		return 0
	}
	return ix.dist2SoA(q, bLo*Dims, bRows, row-bLo)
}

// dist2SoA computes the exact squared distance for a row given its bucket's SoA
// base (bLo*Dims), the bucket's row count, and the row's bucket-local index.
func (ix *Index) dist2SoA(q *[Dims]float64, sBase, bRows, local int) float64 {
	data := ix.data
	var d float64
	for i := 0; i < Dims; i++ {
		diff := q[i] - dequantTab[data[sBase+i*bRows+local]]
		d += diff * diff
	}
	return d
}

// intDist is the integer squared distance Σ (qcode[d]-code[d])² for one row, with
// the query quantized to the native grid. ≈ float_dist*1e8; a cheap approximation
// used to COLLECT candidate rows (the exact 5-NN comes from a float64 refine of the
// collected candidates). Reads the SoA (dim-major) codes for bucket-local row.
func (ix *Index) intDist(qcode *[Dims]int32, sBase, bRows, local int) int32 {
	data := ix.data
	var d int32
	for i := 0; i < Dims; i++ {
		diff := qcode[i] - int32(data[sBase+i*bRows+local])
		d += diff * diff
	}
	return d
}

// scanCellCollect computes the integer distance of every row in a cell (the int16
// SoA kernel for full 8-row blocks, scalar for the <8 tail) and feeds (dist,row)
// into the candidate collector. No float work here — the bulk scan is pure int16
// SIMD; the exact float64 refine runs once, on the few collected candidates.
func (ix *Index) scanCellCollect(qcode *[Dims]int32, sBase, bRows, localLo, gLo, m int, col *intTopK) {
	data := ix.data
	var buf [simdChunk]int32
	done := 0
	for done < m {
		c := m - done
		if c > simdChunk {
			c = simdChunk
		}
		nb := 0
		if useSIMD {
			nb = c / 8 // full 8-row blocks computed by the kernel
			if nb > 0 {
				distSoAi16AVX2(&qcode[0], &data[sBase+localLo+done], bRows, nb, &buf[0])
				for i := 0; i < nb*8; i++ {
					col.consider(buf[i], int32(gLo+done+i))
				}
			}
		}
		// Tail rows (and the whole cell when !useSIMD): scalar int distance.
		for i := nb * 8; i < c; i++ {
			col.consider(ix.intDist(qcode, sBase, bRows, localLo+done+i), int32(gLo+done+i))
		}
		done += c
	}
}

// centroidDist returns the squared distance from q to centroid `cell` (a global
// cell index). Centroids carry their bucket's discrete-dim values, so this
// already includes the bucket penalty for cells in other buckets.
func (ix *Index) centroidDist(q *[Dims]float64, cell int) float64 {
	base := cell * Dims
	c := ix.centroids
	var d float64
	for i := 0; i < Dims; i++ {
		diff := q[i] - c[base+i]
		d += diff * diff
	}
	return d
}

// centroidIntDist is the integer squared distance Σ (qcode[d]-ccode[d])² from the
// quantized query to centroid `local` (bucket-local index) of bucket block `bBlock`
// in the SoA int16 centroid array. The int16 twin of centroidDist, used to pick the
// nprobe cells; it is the scalar tail / fallback for distSoAi16AVX2 (same value).
func (ix *Index) centroidIntDist(qcode *[Dims]int32, bBlock, nlist, local int) int32 {
	cc := ix.centroidsI16
	var d int32
	for i := 0; i < Dims; i++ {
		diff := qcode[i] - int32(cc[bBlock+i*nlist+local])
		d += diff * diff
	}
	return d
}

// insertProbe inserts (d, cell) into the ascending-sorted top-np probe buffers
// (pd distances, pc cell ids) if it beats the current np-th. Strict `<` (first-seen
// wins ties) keeps cell selection deterministic, matching the old float path.
func insertProbe(pd *[maxProbe]int32, pc *[maxProbe]int32, np int, d, cell int32) {
	if d >= pd[np-1] {
		return
	}
	pos := np - 1
	for pos > 0 && pd[pos-1] > d {
		pd[pos] = pd[pos-1]
		pc[pos] = pc[pos-1]
		pos--
	}
	pd[pos] = d
	pc[pos] = cell
}

// Score returns the fraud fraction among the (approximately) K nearest reference
// vectors.
func (ix *Index) Score(query [Dims]float64) float64 {
	if ix.n == 0 {
		return 0
	}
	tk, _, _ := ix.searchTopK(&query)
	return tk.fraudScore()
}

// ScoreCount returns the fraud count among the (approximately) K nearest
// reference vectors (0..K). It runs the same stack-only search path as Score, so
// float64(ScoreCount(q))/K == Score(q). The handler uses the integer count to
// index a precomputed response, keeping the hot path allocation-free.
func (ix *Index) ScoreCount(query [Dims]float64) int {
	if ix.n == 0 {
		return 0
	}
	tk, _, _ := ix.searchTopK(&query)
	n := 0
	for i := 0; i < K; i++ {
		if tk.fraud[i] {
			n++
		}
	}
	return n
}

// ScoreScan is like Score but also returns the number of reference rows visited.
// Diagnostic only.
func (ix *Index) ScoreScan(query [Dims]float64) (float64, int) {
	if ix.n == 0 {
		return 0, 0
	}
	tk, scanned, _ := ix.searchTopK(&query)
	return tk.fraudScore(), scanned
}

// ScoreScanEscalated is ScoreScan plus whether the adaptive high-nprobe pass
// fired for this query (the diag harness uses it to report the escalation rate).
// Diagnostic only.
func (ix *Index) ScoreScanEscalated(query [Dims]float64) (float64, int, bool) {
	if ix.n == 0 {
		return 0, 0, false
	}
	tk, scanned, escalated := ix.searchTopK(&query)
	return tk.fraudScore(), scanned, escalated
}

// ScoreScanTrace is ScoreScan with the cheap-pass vote and per-tier scan counts.
// Diagnostic only.
func (ix *Index) ScoreScanTrace(query [Dims]float64) (float64, int, SearchTrace) {
	if ix.n == 0 {
		return 0, 0, SearchTrace{}
	}
	tk, scanned, trace := ix.searchTopKTrace(&query)
	return tk.fraudScore(), scanned, trace
}

// searchTopK runs the cheap pass at ix.nprobe and, only when the result looks
// like it could be wrong, re-runs the whole search at the higher ix.nprobeHigh.
// The escalation trigger is the 5th-NN search radius (topK.worst()): residual
// misses come exclusively from queries whose true 5-NN crosses a hard-bucket
// boundary, which forces a squared-distance contribution ≥1.0, so a borderline
// result has worst() near/above 1.0. With nprobeHigh disabled (≤nprobe) only the
// cheap pass runs and the result is bit-identical to the single-tier search.
func (ix *Index) searchTopK(q *[Dims]float64) (topK, int, bool) {
	tk, scanned, trace := ix.searchTopKTrace(q)
	return tk, scanned, trace.Escalated
}

func (ix *Index) searchTopKTrace(q *[Dims]float64) (topK, int, SearchTrace) {
	var qcode [Dims]int32
	for i := 0; i < Dims; i++ {
		qcode[i] = int32(quantize(q[i]))
	}

	// Cheap pass: collect candidates from the cells ranked [0, nprobe) of every
	// admissible bucket, then refine to the exact 5-NN for the vote / trigger.
	col := newIntTopK()
	scanned := 0
	ix.scanRange(q, &qcode, 0, ix.nprobe, &col, &scanned, ix.maxScan)
	tk := ix.refine(q, &col)
	cheapCount := fraudCount(&tk)
	trace := SearchTrace{
		CheapFraudCount: cheapCount,
		CheapScanned:    scanned,
	}

	nprobeHigh := ix.highProbeForCount(cheapCount)
	if nprobeHigh > ix.nprobe && ix.shouldEscalate(&tk) {
		// Escalate INCREMENTALLY: extend the SAME candidate collector with the new
		// cells ranked [nprobe, nprobeHigh). The cheap cells [0, nprobe) are never
		// re-scanned, so the candidate set equals a fresh nprobeHigh scan (identical
		// 5-NN and detection) but the work excludes the already-scanned cheap cells.
		// maxScan is disabled here — escalation is the "spend more" path and must
		// never be truncated.
		highScanned := 0
		ix.scanRange(q, &qcode, ix.nprobe, nprobeHigh, &col, &highScanned, 0)
		tk = ix.refine(q, &col)
		trace.Escalated = true
		trace.HighScanned = highScanned
		return tk, scanned + highScanned, trace
	}
	return tk, scanned, trace
}

func (ix *Index) highProbeForCount(count int) int {
	if count >= 0 && count <= K && ix.nprobeHighByCount[count] > 0 {
		return ix.nprobeHighByCount[count]
	}
	return ix.nprobeHigh
}

// shouldEscalate decides whether the cheap pass warrants the high-nprobe re-run.
// Primary signal (triggerMargin>=0): the fraud vote sits within triggerMargin of
// the 0.6 decision boundary (K*Threshold = 3 of 5) — the only queries whose exact
// 5-NN could flip the approve/deny outcome, so the only ones where a residual
// miss costs a failure. Measured on the real set: margin=1 (count 2..4) → E=0 at
// ~3% escalation. The 5th-NN radius alone is a poor trigger (the approximate pass
// inflates worst(), firing on ~34% while still missing real flips), so it is only
// the fallback when the margin gate is disabled (triggerMargin<0).
func (ix *Index) shouldEscalate(tk *topK) bool {
	if ix.triggerMargin >= 0 {
		return nearVote(tk, ix.triggerMargin)
	}
	return tk.worst() >= ix.triggerRadius
}

// scanRange probes the home bucket and the admissible cross-buckets, scanning cells
// ranked [npLo, npHi) of each into the shared candidate collector col, and adds the
// rows visited to *scanned. Stack-only / zero-allocation; callers pass the
// pre-quantized qcode so the two-tier dispatcher quantizes once.
//
// The cross-bucket prune is a conservative real-distance lower bound (scaled to
// integer ~*1e8): it skips a bucket only when its forced penalty exceeds the worst
// collected candidate by a safe margin (>= 1e5 covers the query-grid rounding
// |d_int-d_float*1e8| <= sqrt(14*d_float)*1e4). Because the test never drops a bucket
// that could hold a true top-candK row, the collected set — and the exact 5-NN
// refined from it — is independent of probe order. That is what makes incremental
// escalation safe: extending an existing col with the [nprobe, nprobeHigh) cells
// yields the same candidates as a fresh [0, nprobeHigh) scan.
func (ix *Index) scanRange(q *[Dims]float64, qcode *[Dims]int32, npLo, npHi int, col *intTopK, scanned *int, capScan int) {
	b0 := bucketOf(q)
	ix.probeBucketRange(q, qcode, b0, npLo, npHi, col, scanned, capScan)
	for b := 0; b < numBuckets; b++ {
		if b == b0 {
			continue
		}
		if capScan > 0 && *scanned >= capScan {
			break
		}
		if col.full() && bucketPenalty(q, b)*1e8 >= float64(col.worst())+1e5 {
			continue
		}
		ix.probeBucketRange(q, qcode, b, npLo, npHi, col, scanned, capScan)
	}
}

// refine turns the collected integer-distance candidates into the exact float64 5-NN.
func (ix *Index) refine(q *[Dims]float64, col *intTopK) topK {
	tk := newTopK()
	for i := 0; i < col.n; i++ {
		row := int(col.row[i])
		tk.consider(ix.dist2(q, row), ix.isFraud(row))
	}
	return tk
}

// nearVote reports whether the cheap pass's fraud vote is within `margin` of the
// 0.6 decision boundary (K*Threshold = 3 of 5). Optional AND-gate for escalation.
func nearVote(tk *topK, margin float64) bool {
	n := fraudCount(tk)
	d := float64(n) - K*Threshold
	if d < 0 {
		d = -d
	}
	return d <= margin
}

func fraudCount(tk *topK) int {
	n := 0
	for i := 0; i < K; i++ {
		if tk.fraud[i] {
			n++
		}
	}
	return n
}

// probeBucketRange scans cells ranked [npLo, npHi) (by ascending centroid distance)
// of bucket b — i.e. it selects the npHi nearest cells but scans only those whose
// rank is >= npLo. Cell selection touches only the nlist centroids (contiguous); the
// chosen cells are contiguous row runs, so the candidate scan is prefetch-friendly.
//
// Two-tier reuse: the cheap pass calls probeBucketRange(b, 0, nprobe); the escalated
// pass calls probeBucketRange(b, nprobe, nprobeHigh) into the SAME candidate
// collector. The nprobe nearest cells are a deterministic prefix of the npHi nearest
// (insertProbe is order-stable), so the cheap pass already scanned ranks [0, nprobe)
// and the escalation never re-scans them — it only adds the genuinely new cells
// [nprobe, nprobeHigh). The final candidate set is identical to a single fresh scan
// of the npHi nearest cells, so the exact 5-NN (and detection) is unchanged.
func (ix *Index) probeBucketRange(q *[Dims]float64, qcode *[Dims]int32, b, npLo, npHi int, col *intTopK, scanned *int, capScan int) {
	nlist := ix.nlist
	base := b * nlist
	np := npHi
	if np > nlist {
		np = nlist
	}
	if np > maxProbe {
		np = maxProbe
	}
	if np < 1 {
		np = 1
	}
	if npLo < 0 {
		npLo = 0
	}
	if npLo >= np {
		return // nothing new to scan in this range
	}

	// Select the np nearest centroids by INTEGER (query-quantized) distance into stack
	// arrays (no allocation). The int16 SIMD kernel scans the nlist centroids 8 at a
	// time (SoA dim-major per bucket); the <8 tail (and the whole scan when !useSIMD)
	// uses the scalar centroidIntDist twin. Cell selection only needs ordering, so the
	// integer distance suffices — the exact 5-NN still comes from the float64 refine.
	var pd [maxProbe]int32
	var pc [maxProbe]int32
	for i := 0; i < np; i++ {
		pd[i] = math.MaxInt32
	}
	bBlock := b * Dims * nlist // start of bucket b's SoA centroid block
	var cbuf [simdChunk]int32
	done := 0
	for done < nlist {
		c := nlist - done
		if c > simdChunk {
			c = simdChunk
		}
		nb := 0
		if useSIMD {
			nb = c / 8 // full 8-centroid blocks computed by the kernel
			if nb > 0 {
				distSoAi16AVX2(&qcode[0], &ix.centroidsI16[bBlock+done], nlist, nb, &cbuf[0])
				for i := 0; i < nb*8; i++ {
					insertProbe(&pd, &pc, np, cbuf[i], int32(base+done+i))
				}
			}
		}
		// Tail centroids (and the whole scan when !useSIMD): scalar int distance.
		for i := nb * 8; i < c; i++ {
			insertProbe(&pd, &pc, np, ix.centroidIntDist(qcode, bBlock, nlist, done+i), int32(base+done+i))
		}
		done += c
	}

	bLo := int(ix.bucketStart[b])
	bRows := int(ix.bucketStart[b+1]) - bLo
	sBase := bLo * Dims
	for i := npLo; i < np; i++ {
		if pd[i] == math.MaxInt32 {
			break // fewer non-empty candidates than np
		}
		cell := int(pc[i])
		lo, hi := int(ix.cellStart[cell]), int(ix.cellStart[cell+1])
		m := hi - lo
		if m <= 0 {
			continue
		}
		ix.scanCellCollect(qcode, sBase, bRows, lo-bLo, lo, m, col)
		*scanned += m
		if capScan > 0 && *scanned >= capScan {
			return
		}
	}
}

// BruteScore is the exact O(n) reference: it scans every stored row. The offline
// harness uses it as the ground-truth oracle. Not on the serving path.
func (ix *Index) BruteScore(query [Dims]float64) float64 {
	s, _ := ix.BruteDebug(query)
	return s
}

// BruteDebug returns the exact fraud score AND the squared distance to the K-th
// nearest reference (the search radius). Diagnostic only.
func (ix *Index) BruteDebug(query [Dims]float64) (score, kthDist2 float64) {
	if ix.n == 0 {
		return 0, 0
	}
	tk := newTopK()
	for r := 0; r < ix.n; r++ {
		tk.consider(ix.dist2(&query, r), ix.isFraud(r))
	}
	return tk.fraudScore(), tk.worst()
}

// --- topK: sorted buffer of the K nearest seen so far --------------------

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

// worst is the distance of the current K-th nearest.
func (t *topK) worst() float64 { return t.dist[K-1] }

// intTopK collects the candK rows of smallest INTEGER distance during the scan
// (sorted ascending, stack-resident, zero-allocation). Its worst() drives the
// conservative cross-bucket prune; its rows are the candidates the float64 refine
// turns into the exact 5-NN.
type intTopK struct {
	dist [candK]int32
	row  [candK]int32
	n    int
}

func newIntTopK() intTopK {
	var t intTopK
	for i := range t.dist {
		t.dist[i] = math.MaxInt32
	}
	return t
}

func (t *intTopK) full() bool   { return t.n >= candK }
func (t *intTopK) worst() int32 { return t.dist[candK-1] }

// consider inserts (d, row) if it beats the current candK-th smallest. Strict `<`
// (first-seen wins ties) so the candidate set is deterministic.
func (t *intTopK) consider(d, row int32) {
	if d >= t.dist[candK-1] {
		return
	}
	pos := candK - 1
	for pos > 0 && t.dist[pos-1] > d {
		t.dist[pos] = t.dist[pos-1]
		t.row[pos] = t.row[pos-1]
		pos--
	}
	t.dist[pos] = d
	t.row[pos] = row
	if t.n < candK {
		t.n++
	}
}

// consider inserts (dist, fraud) if it beats the current K-th. The strict `<`
// (first-seen wins ties) matches the float64 ground-truth oracle.
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
	n := 0
	for i := 0; i < K; i++ {
		if t.fraud[i] {
			n++
		}
	}
	return float64(n) / float64(K)
}
