// Package model evaluates a gradient-boosted tree ensemble (trained offline by
// train/train.py with xgboost, objective binary:logistic) in PURE Go — no deps,
// no scan over the dataset. It is the Caminho C scorer: a query is classified by
// walking ~100 small trees (a few hundred comparisons) instead of a K-NN search
// over millions of vectors, so inference is sub-millisecond.
//
// The model.json artifact (produced offline, embedded at build) holds the trees
// in xgboost's dump_format="json". Score reproduces xgboost's prediction:
//
//	P(fraud) = sigmoid( logit(base_score) + Σ_tree leaf_reached(tree) )
//
// A query is approved when P(fraud) < Tau (the calibrated decision threshold).
package model

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"rinha-fraud/internal/vectorize"
)

// Model is the parsed ensemble.
type Model struct {
	baseMargin float64 // logit(base_score); the ensemble's initial margin
	tau        float64 // approved when Score(q) < tau
	trees      []tree
}

// tree is one boosted tree, flattened by xgboost nodeid for pointer-free walks.
// Internal node i: feature[i] >= 0; branch to yes[i] when x[feature] < threshold,
// else no[i]. Leaf node i: feature[i] == leafMarker; its value is leaf[i].
type tree struct {
	feature   []int32
	threshold []float32 // float32 to match xgboost's internal split comparison exactly
	yes       []int32
	no        []int32
	leaf      []float64
}

const leafMarker = -1

// rawModel mirrors resources/model.json (written by train/train.py).
type rawModel struct {
	BaseScore float64   `json:"base_score"`
	Tau       float64   `json:"tau"`
	NFeatures int       `json:"n_features"`
	NTrees    int       `json:"n_trees"`
	Trees     []rawNode `json:"trees"`
}

// rawNode mirrors xgboost dump_format="json": leaf nodes carry "leaf"; internal
// nodes carry "split" ("f<idx>"), "split_condition", "yes"/"no" and nested
// "children".
type rawNode struct {
	NodeID    int       `json:"nodeid"`
	Split     string    `json:"split"`
	SplitCond float64   `json:"split_condition"`
	Yes       int       `json:"yes"`
	No        int       `json:"no"`
	Leaf      *float64  `json:"leaf"`
	Children  []rawNode `json:"children"`
}

// LoadModel parses model.json bytes (embedded by main at build time).
func LoadModel(data []byte) (*Model, error) {
	var rm rawModel
	if err := json.Unmarshal(data, &rm); err != nil {
		return nil, fmt.Errorf("model: parse: %w", err)
	}
	if rm.NFeatures != vectorize.Dims {
		return nil, fmt.Errorf("model: n_features %d != %d", rm.NFeatures, vectorize.Dims)
	}
	if len(rm.Trees) == 0 {
		return nil, fmt.Errorf("model: no trees")
	}
	if rm.BaseScore <= 0 || rm.BaseScore >= 1 {
		return nil, fmt.Errorf("model: base_score %v outside (0,1)", rm.BaseScore)
	}
	m := &Model{
		baseMargin: math.Log(rm.BaseScore / (1 - rm.BaseScore)),
		tau:        rm.Tau,
		trees:      make([]tree, len(rm.Trees)),
	}
	for i := range rm.Trees {
		t, err := buildTree(&rm.Trees[i])
		if err != nil {
			return nil, fmt.Errorf("model: tree %d: %w", i, err)
		}
		m.trees[i] = t
	}
	return m, nil
}

// buildTree flattens a parsed xgboost tree into arrays indexed by nodeid.
// xgboost nodeids follow heap numbering with gaps from pruned branches; the
// arrays are sized to maxID+1 and gaps stay as unreachable leaves.
func buildTree(root *rawNode) (tree, error) {
	nodes := map[int]*rawNode{}
	maxID := 0
	var collect func(n *rawNode)
	collect = func(n *rawNode) {
		nodes[n.NodeID] = n
		if n.NodeID > maxID {
			maxID = n.NodeID
		}
		for i := range n.Children {
			collect(&n.Children[i])
		}
	}
	collect(root)

	size := maxID + 1
	t := tree{
		feature:   make([]int32, size),
		threshold: make([]float32, size),
		yes:       make([]int32, size),
		no:        make([]int32, size),
		leaf:      make([]float64, size),
	}
	for id := range t.feature {
		t.feature[id] = leafMarker
	}
	for id, n := range nodes {
		if n.Leaf != nil {
			t.feature[id] = leafMarker
			t.leaf[id] = *n.Leaf
			continue
		}
		if len(n.Split) < 2 || n.Split[0] != 'f' {
			return tree{}, fmt.Errorf("bad split %q", n.Split)
		}
		f, err := strconv.Atoi(n.Split[1:])
		if err != nil {
			return tree{}, fmt.Errorf("bad split feature %q: %w", n.Split, err)
		}
		if f < 0 || f >= vectorize.Dims {
			return tree{}, fmt.Errorf("split feature %d out of range", f)
		}
		t.feature[id] = int32(f)
		t.threshold[id] = float32(n.SplitCond)
		t.yes[id] = int32(n.Yes)
		t.no[id] = int32(n.No)
	}
	return t, nil
}

// Score returns P(fraud) = sigmoid(baseMargin + Σ leaf values reached).
func (m *Model) Score(q [vectorize.Dims]float64) float64 {
	margin := m.baseMargin
	for ti := range m.trees {
		t := &m.trees[ti]
		id := int32(0)
		for t.feature[id] != leafMarker {
			if float32(q[t.feature[id]]) < t.threshold[id] {
				id = t.yes[id]
			} else {
				id = t.no[id]
			}
		}
		margin += t.leaf[id]
	}
	return 1 / (1 + math.Exp(-margin))
}

// Tau is the decision threshold: a transaction is approved when Score(q) < Tau.
func (m *Model) Tau() float64 { return m.tau }
