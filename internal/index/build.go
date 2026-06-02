package index

import (
	"runtime"
	"sync"
)

// Default build parameters.
const (
	DefaultNList       = 1024 // k-means cells per bucket
	DefaultKMeansIters = 10
	kmeansSample       = 50000 // rows sampled to train k-means per bucket (bounds build cost)
)

// Builder accumulates reference vectors (quantized on insertion to keep memory
// bounded) and produces an Index. Construction (offline, full CPU):
//
//  1. counting-sort rows into the 16 hard buckets,
//  2. k-means each bucket into nlist cells (trained on a sample, then all rows
//     assigned), and
//  3. counting-sort rows within each bucket into contiguous cell runs.
type Builder struct {
	data   []uint16 // n*Dims quantized, insertion order
	bucket []uint8  // n, hard-bucket index (0..15)
	fraud  []bool   // n
	n      int

	nlist int
	iters int
}

// NewBuilder pre-allocates for capacityHint rows. nlist cells per bucket, iters
// k-means iterations (≤0 → defaults; tests pass nlist=1 for an exact per-bucket
// index).
func NewBuilder(capacityHint, nlist, iters int) *Builder {
	if capacityHint < 0 {
		capacityHint = 0
	}
	if nlist <= 0 {
		nlist = DefaultNList
	}
	if iters <= 0 {
		iters = DefaultKMeansIters
	}
	return &Builder{
		data:   make([]uint16, 0, capacityHint*Dims),
		bucket: make([]uint8, 0, capacityHint),
		fraud:  make([]bool, 0, capacityHint),
		nlist:  nlist,
		iters:  iters,
	}
}

// Add quantizes vec and records its hard bucket and fraud label.
func (b *Builder) Add(vec [Dims]float64, fraud bool) {
	for i := 0; i < Dims; i++ {
		b.data = append(b.data, quantize(vec[i]))
	}
	b.bucket = append(b.bucket, uint8(bucketOf(&vec)))
	b.fraud = append(b.fraud, fraud)
	b.n++
}

// Len reports how many vectors have been added.
func (b *Builder) Len() int { return b.n }

// Build produces the IVF index. The returned Index has nprobe defaulted to a
// scan of all cells (exact within a bucket); production lowers it via SetNProbe.
func (b *Builder) Build() *Index {
	nlist := b.nlist
	defProbe := nlist // exact-within-bucket by default; SetNProbe trims it
	if defProbe > maxProbe {
		defProbe = maxProbe
	}
	ix := &Index{
		n:         b.n,
		nlist:     nlist,
		centroids: make([]float64, numBuckets*nlist*Dims),
		cellStart: make([]int32, numBuckets*nlist+1),
		nprobe:    defProbe,
	}

	// 1. Bucket histogram → contiguous bucket ranges.
	var counts [numBuckets]int32
	for _, bk := range b.bucket {
		counts[bk]++
	}
	var acc int32
	for i := 0; i < numBuckets; i++ {
		ix.bucketStart[i] = acc
		acc += counts[i]
	}
	ix.bucketStart[numBuckets] = acc

	// 2. Scatter rows into bucket order.
	bdata := make([]uint16, b.n*Dims)
	bfraud := make([]bool, b.n)
	pos := make([]int32, numBuckets)
	copy(pos, ix.bucketStart[:numBuckets])
	for row := 0; row < b.n; row++ {
		bk := b.bucket[row]
		dst := pos[bk]
		pos[bk]++
		copy(bdata[int(dst)*Dims:int(dst)*Dims+Dims], b.data[row*Dims:row*Dims+Dims])
		bfraud[dst] = b.fraud[row]
	}

	// 3. Per bucket: k-means, then counting-sort rows into contiguous cells.
	finalData := make([]uint16, b.n*Dims)
	finalFraud := make([]bool, b.n)
	for bk := 0; bk < numBuckets; bk++ {
		lo, hi := int(ix.bucketStart[bk]), int(ix.bucketStart[bk+1])
		cbase := bk * nlist
		km := &kmeans{data: bdata, dims: Dims, nlist: nlist}
		assign := km.run(lo, hi, ix.centroids[cbase*Dims:(cbase+nlist)*Dims], b.iters)

		// Cell histogram within the bucket → CSR offsets (global indices).
		cellCounts := make([]int32, nlist)
		for _, c := range assign {
			cellCounts[c]++
		}
		cellPos := make([]int32, nlist)
		off := int32(lo)
		for c := 0; c < nlist; c++ {
			ix.cellStart[cbase+c] = off
			cellPos[c] = off
			off += cellCounts[c]
		}
		// Scatter the bucket's rows into cell order.
		for i := 0; i < hi-lo; i++ {
			c := assign[i]
			dst := cellPos[c]
			cellPos[c]++
			src := lo + i
			copy(finalData[int(dst)*Dims:int(dst)*Dims+Dims], bdata[src*Dims:src*Dims+Dims])
			finalFraud[dst] = bfraud[src]
		}
	}
	ix.cellStart[numBuckets*nlist] = int32(b.n)

	// 4. Pack fraud bits over the final row order.
	ix.fraud = make([]uint64, (b.n+63)/64)
	for r := 0; r < b.n; r++ {
		if finalFraud[r] {
			ix.fraud[r>>6] |= 1 << uint(r&63)
		}
	}

	// 5. Transpose each bucket to SoA (dim-major) for the SIMD row kernel. Within a
	//    bucket's row range, store all rows' dim 0, then all rows' dim 1, ... so the
	//    kernel can load 8 consecutive rows' dim-d codes in one m128. Cells stay
	//    contiguous local-row ranges within each dim segment (cellStart unchanged,
	//    still global rows). Row identity / fraud bits / centroids are unchanged.
	//    +simdTailPad uint16 of zero tail so the kernel's last m128 load can't OOB.
	soa := make([]uint16, b.n*Dims+simdTailPad)
	for bk := 0; bk < numBuckets; bk++ {
		lo, hi := int(ix.bucketStart[bk]), int(ix.bucketStart[bk+1])
		bRows := hi - lo
		if bRows == 0 {
			continue
		}
		bBase := lo * Dims
		for local := 0; local < bRows; local++ {
			off := (lo + local) * Dims
			for d := 0; d < Dims; d++ {
				soa[bBase+d*bRows+local] = finalData[off+d]
			}
		}
	}
	ix.data = soa

	// 6. Quantized SoA twin of the centroids for the int16 SIMD cell-selection scan.
	ix.buildCentroidI16()
	return ix
}

// buildCentroidI16 fills centroidsI16 from the float64 centroids: per bucket, store
// the quantized centroid codes dim-major (centroidsI16[b*Dims*nlist + d*nlist + c])
// so distSoAi16AVX2 can load 8 cells of one dim per m128. Derived (not serialized),
// so it runs both after Build and after a deserialize load. quantize() reuses the
// row quantization (value space → native grid; mean -1 of a null dim → 0 sentinel),
// keeping centroid codes on the same grid as the stored row codes.
func (ix *Index) buildCentroidI16() {
	nlist := ix.nlist
	ix.centroidsI16 = make([]uint16, numBuckets*nlist*Dims+simdTailPad)
	for b := 0; b < numBuckets; b++ {
		bBlock := b * Dims * nlist
		cbase := b * nlist
		for c := 0; c < nlist; c++ {
			off := (cbase + c) * Dims
			for d := 0; d < Dims; d++ {
				ix.centroidsI16[bBlock+d*nlist+c] = quantize(ix.centroids[off+d])
			}
		}
	}
}

// kmeans runs Lloyd's algorithm over a bucket's rows in the dequantized value
// space, training on an evenly-spaced sample (deterministic, reproducible) and
// assigning every row at the end.
type kmeans struct {
	data  []uint16
	dims  int
	nlist int
}

// run fills centroids (nlist*dims) for rows [lo,hi) and returns each row's cell
// assignment (length hi-lo, values 0..nlist-1).
func (k *kmeans) run(lo, hi int, centroids []float64, iters int) []int32 {
	m := hi - lo
	assign := make([]int32, m)
	if m == 0 {
		return assign
	}
	nlist := k.nlist
	if nlist > m {
		nlist = m // never more cells than rows
	}
	D := k.dims

	// Seed centroids from evenly-spaced rows.
	stride := m / nlist
	if stride < 1 {
		stride = 1
	}
	for c := 0; c < nlist; c++ {
		src := lo + c*stride
		if src >= hi {
			src = hi - 1
		}
		for d := 0; d < D; d++ {
			centroids[c*D+d] = dequant(k.data[src*D+d])
		}
	}

	// Train on a sample, then assign all rows.
	sample := sampleRows(lo, hi, kmeansSample)
	sum := make([]float64, nlist*D)
	cnt := make([]int32, nlist)
	for it := 0; it < iters; it++ {
		for i := range sum {
			sum[i] = 0
		}
		for i := range cnt {
			cnt[i] = 0
		}
		for _, row := range sample {
			c := k.nearest(row, centroids, nlist)
			base := c * D
			off := row * D
			for d := 0; d < D; d++ {
				sum[base+d] += dequant(k.data[off+d])
			}
			cnt[c]++
		}
		for c := 0; c < nlist; c++ {
			if cnt[c] == 0 {
				continue
			}
			base := c * D
			inv := 1 / float64(cnt[c])
			for d := 0; d < D; d++ {
				centroids[base+d] = sum[base+d] * inv
			}
		}
	}

	// Final assignment of every row, in parallel (deterministic split).
	parallelFor(m, func(p0, p1 int) {
		for i := p0; i < p1; i++ {
			assign[i] = int32(k.nearest(lo+i, centroids, nlist))
		}
	})
	return assign
}

// nearest returns the index of the centroid closest to row (dequantized).
func (k *kmeans) nearest(row int, centroids []float64, nlist int) int {
	D := k.dims
	off := row * D
	best, bestD := 0, 0.0
	first := true
	for c := 0; c < nlist; c++ {
		base := c * D
		var d float64
		for j := 0; j < D; j++ {
			diff := dequant(k.data[off+j]) - centroids[base+j]
			d += diff * diff
		}
		if first || d < bestD {
			best, bestD, first = c, d, false
		}
	}
	return best
}

// sampleRows returns up to `size` evenly-spaced row indices in [lo,hi).
func sampleRows(lo, hi, size int) []int {
	m := hi - lo
	if size >= m {
		ids := make([]int, m)
		for i := range ids {
			ids[i] = lo + i
		}
		return ids
	}
	stride := m / size
	ids := make([]int, size)
	for i := 0; i < size; i++ {
		ids[i] = lo + i*stride
	}
	return ids
}

// parallelFor splits [0,n) into one contiguous chunk per worker and runs fn on
// each concurrently. Deterministic split → reproducible. Single-threaded for
// small n or one core.
func parallelFor(n int, fn func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || n < 8192 {
		fn(0, n)
		return
	}
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		p0 := w * chunk
		if p0 >= n {
			break
		}
		p1 := p0 + chunk
		if p1 > n {
			p1 = n
		}
		wg.Add(1)
		go func(p0, p1 int) {
			defer wg.Done()
			fn(p0, p1)
		}(p0, p1)
	}
	wg.Wait()
}
