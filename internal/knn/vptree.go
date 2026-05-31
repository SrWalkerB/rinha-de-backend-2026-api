package knn

import (
	"math"
	"sort"

	"rinha-fraud/internal/vectorize"
)

// vpTree is a vantage-point tree over the quantized rows. It answers exact K-NN
// queries identical to brute force, but prunes whole subtrees using the triangle
// inequality, so a query visits far fewer than n rows.
//
// Each node stores a vantage point (a row index) and a radius (the median true
// distance from the vantage point to its descendants). The "inner" subtree holds
// points within the radius; "outer" holds the rest.
type vpTree struct {
	nodes []vpNode
	root  int32
}

type vpNode struct {
	point  int32   // row index of the vantage point
	radius float32 // true (euclidean) distance threshold splitting inner/outer
	inner  int32   // child node index, or -1
	outer  int32   // child node index, or -1
}

type vpPair struct {
	id int32
	d2 uint64
}

type vpBuilder struct {
	ix      *Index
	t       *vpTree
	ids     []int32 // working permutation of row indices, partitioned in place
	scratch []vpPair
}

// buildVPTree constructs the tree over all rows of ix.
func buildVPTree(ix *Index) *vpTree {
	b := &vpBuilder{
		ix:      ix,
		t:       &vpTree{nodes: make([]vpNode, 0, ix.n)},
		ids:     make([]int32, ix.n),
		scratch: make([]vpPair, ix.n),
	}
	for i := range b.ids {
		b.ids[i] = int32(i)
	}
	b.t.root = b.build(0, ix.n)
	b.scratch = nil // free the build-only buffer
	return b.t
}

// build constructs a subtree over ids[lo:hi] and returns its node index.
func (b *vpBuilder) build(lo, hi int) int32 {
	if lo >= hi {
		return -1
	}
	ni := int32(len(b.t.nodes))
	vp := b.ids[lo]
	b.t.nodes = append(b.t.nodes, vpNode{point: vp, inner: -1, outer: -1})
	if hi-lo == 1 {
		return ni // leaf
	}

	// Distances from the vantage point to the rest of the range.
	m := hi - lo - 1
	sc := b.scratch[:m]
	for k := 0; k < m; k++ {
		id := b.ids[lo+1+k]
		sc[k] = vpPair{id: id, d2: b.ix.rowDist2(int(vp), int(id))}
	}
	sort.Slice(sc, func(i, j int) bool { return sc[i].d2 < sc[j].d2 })

	mid := m / 2
	medianD2 := sc[mid].d2
	for k := 0; k < m; k++ { // write the partitioned order back
		b.ids[lo+1+k] = sc[k].id
	}

	// Recurse first (append children), then fix this node's fields — appending
	// to b.t.nodes may reallocate, so never cache a *vpNode across recursion.
	radius := float32(math.Sqrt(float64(medianD2)))
	inner := b.build(lo+1, lo+1+mid)
	outer := b.build(lo+1+mid, hi)
	b.t.nodes[ni].radius = radius
	b.t.nodes[ni].inner = inner
	b.t.nodes[ni].outer = outer
	return ni
}

func (t *vpTree) search(ix *Index, q *[vectorize.Dims]uint16) float64 {
	tk := t.searchTopK(ix, q)
	return tk.fraudScore()
}

func (t *vpTree) searchTopK(ix *Index, q *[vectorize.Dims]uint16) topK {
	tk := newTopK()
	t.searchNode(ix, q, t.root, &tk)
	return tk
}

func (t *vpTree) searchNode(ix *Index, q *[vectorize.Dims]uint16, ni int32, tk *topK) {
	if ni < 0 {
		return
	}
	point := t.nodes[ni].point
	d2 := ix.dist2(q, int(point))
	tk.consider(d2, ix.isFraud(int(point)))

	inner, outer := t.nodes[ni].inner, t.nodes[ni].outer
	if inner < 0 && outer < 0 {
		return
	}

	// Pruning uses TRUE distances (the triangle inequality does not hold for
	// squared distances). These sqrt calls run only on visited nodes (few).
	d := math.Sqrt(float64(d2))
	r := float64(t.nodes[ni].radius)

	if d < r {
		// Query is inside the radius: the inner subtree is the closer one.
		t.searchNode(ix, q, inner, tk)
		tau := math.Sqrt(float64(tk.worst())) // current K-th distance (search radius)
		if d+tau >= r {
			t.searchNode(ix, q, outer, tk)
		}
	} else {
		t.searchNode(ix, q, outer, tk)
		tau := math.Sqrt(float64(tk.worst()))
		if d-tau <= r {
			t.searchNode(ix, q, inner, tk)
		}
	}
}
