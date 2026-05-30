package knn

import (
	"math"
	"runtime"
	"sync"

	"rinha-fraud/internal/vectorize"
)

// maxProbe caps NProbe so the per-query probe buffer can live on the stack
// (zero allocations per Score). NProbe is clamped to this at build time.
const maxProbe = 64

// ivfIndex is an Inverted File index: k-means partitions the rows into nlist
// cells; a query scans only the nprobe cells whose centroid is nearest. This is
// APPROXIMATE — the true K-NN may sit in an unscanned cell — trading a little
// recall for a large speedup. Higher nprobe => better recall, slower query.
type ivfIndex struct {
	centroids []uint8   // nlist * Dims
	lists     [][]int32 // row ids per cell
	nlist     int
	nprobe    int
}

func buildIVF(ix *Index, cfg BuildConfig) *ivfIndex {
	D := vectorize.Dims

	nlist := cfg.NList
	if nlist <= 0 {
		nlist = 256
	}
	if nlist > ix.n {
		nlist = ix.n
	}
	nprobe := cfg.NProbe
	if nprobe <= 0 {
		nprobe = 8
	}
	if nprobe > nlist {
		nprobe = nlist
	}
	if nprobe > maxProbe {
		nprobe = maxProbe
	}
	iters := cfg.Iters
	if iters <= 0 {
		iters = 8
	}

	f := &ivfIndex{
		centroids: make([]uint8, nlist*D),
		lists:     make([][]int32, nlist),
		nlist:     nlist,
		nprobe:    nprobe,
	}

	// Seed centroids from evenly-spaced rows (deterministic, spreads them out).
	stride := ix.n / nlist
	if stride < 1 {
		stride = 1
	}
	for c := 0; c < nlist; c++ {
		row := c * stride
		copy(f.centroids[c*D:(c+1)*D], ix.data[row*D:(row+1)*D])
	}

	// Train on a sample (bounds build cost), then assign every row to a cell.
	f.trainKMeans(ix, sampleRows(ix.n, 50_000), iters)
	f.assignAll(ix)
	return f
}

// sampleRows returns up to `size` evenly-spaced row indices.
func sampleRows(n, size int) []int32 {
	if size >= n {
		ids := make([]int32, n)
		for i := range ids {
			ids[i] = int32(i)
		}
		return ids
	}
	stride := n / size
	ids := make([]int32, size)
	for i := 0; i < size; i++ {
		ids[i] = int32(i * stride)
	}
	return ids
}

func (f *ivfIndex) trainKMeans(ix *Index, sample []int32, iters int) {
	D := vectorize.Dims
	sum := make([]float64, f.nlist*D)
	cnt := make([]int, f.nlist)
	assign := make([]int32, len(sample))
	for it := 0; it < iters; it++ {
		// Assignment step is the costly part (each point scans all centroids) —
		// run it across cores. Each index is written once, so no locking. Range
		// split is deterministic, so the trained centroids are reproducible.
		parallelFor(len(sample), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				assign[i] = int32(f.nearestCentroid(ix, int(sample[i])))
			}
		})

		for i := range sum {
			sum[i] = 0
		}
		for i := range cnt {
			cnt[i] = 0
		}
		for i, row := range sample {
			c := int(assign[i])
			off := int(row) * D
			base := c * D
			for d := 0; d < D; d++ {
				sum[base+d] += float64(ix.data[off+d])
			}
			cnt[c]++
		}
		for c := 0; c < f.nlist; c++ {
			if cnt[c] == 0 {
				continue // keep the old centroid for an empty cell
			}
			base := c * D
			for d := 0; d < D; d++ {
				f.centroids[base+d] = uint8(math.Round(sum[base+d] / float64(cnt[c])))
			}
		}
	}
}

func (f *ivfIndex) assignAll(ix *Index) {
	// Assign every row to its nearest centroid in parallel (independent writes),
	// then pack the cell lists into ONE contiguous backing array — fewer, larger
	// allocations than appending to nlist separate slices, and the same layout
	// the serializer writes.
	assign := make([]int32, ix.n)
	parallelFor(ix.n, func(lo, hi int) {
		for row := lo; row < hi; row++ {
			assign[row] = int32(f.nearestCentroid(ix, row))
		}
	})

	counts := make([]int, f.nlist)
	for _, c := range assign {
		counts[c]++
	}
	flat := make([]int32, ix.n)
	off := 0
	for c := 0; c < f.nlist; c++ {
		f.lists[c] = flat[off : off : off+counts[c]] // len 0, cap counts[c]
		off += counts[c]
	}
	for row := 0; row < ix.n; row++ {
		c := assign[row]
		f.lists[c] = append(f.lists[c], int32(row))
	}
}

// parallelFor splits [0,n) into contiguous chunks (one per GOMAXPROCS worker)
// and runs fn on each concurrently. The split is deterministic. Falls back to a
// direct call when there is one core or little work (e.g. the runtime fallback
// build runs under GOMAXPROCS=1).
func parallelFor(n int, fn func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || n < 4096 {
		fn(0, n)
		return
	}
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= n {
			break
		}
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// nearestCentroid returns the index of the centroid closest to row.
func (f *ivfIndex) nearestCentroid(ix *Index, row int) int {
	D := vectorize.Dims
	off := row * D
	data := ix.data
	best := 0
	var bestD uint32 = math.MaxUint32
	for c := 0; c < f.nlist; c++ {
		base := c * D
		var dist uint32
		for d := 0; d < D; d++ {
			diff := int32(data[off+d]) - int32(f.centroids[base+d])
			dist += uint32(diff * diff)
		}
		if dist < bestD {
			bestD = dist
			best = c
		}
	}
	return best
}

// centroidDist returns the squared distance from query q to centroid c.
func (f *ivfIndex) centroidDist(q *[vectorize.Dims]uint8, c int) uint32 {
	D := vectorize.Dims
	base := c * D
	var dist uint32
	for d := 0; d < D; d++ {
		diff := int32(q[d]) - int32(f.centroids[base+d])
		dist += uint32(diff * diff)
	}
	return dist
}

func (f *ivfIndex) search(ix *Index, q *[vectorize.Dims]uint8) float64 {
	tk := f.searchTopK(ix, q)
	return tk.fraudScore()
}

func (f *ivfIndex) searchTopK(ix *Index, q *[vectorize.Dims]uint8) topK {
	np := f.nprobe

	// Select the nprobe nearest centroids into stack arrays (no allocation).
	var pd [maxProbe]uint32
	var pc [maxProbe]int32
	for i := 0; i < np; i++ {
		pd[i] = math.MaxUint32
	}
	for c := 0; c < f.nlist; c++ {
		dist := f.centroidDist(q, c)
		if dist >= pd[np-1] {
			continue
		}
		pos := np - 1
		for pos > 0 && pd[pos-1] > dist {
			pd[pos] = pd[pos-1]
			pc[pos] = pc[pos-1]
			pos--
		}
		pd[pos] = dist
		pc[pos] = int32(c)
	}

	// Scan only the chosen cells.
	tk := newTopK()
	for i := 0; i < np; i++ {
		for _, row := range f.lists[pc[i]] {
			tk.consider(ix.dist2(q, int(row)), ix.isFraud(int(row)))
		}
	}
	return tk
}
