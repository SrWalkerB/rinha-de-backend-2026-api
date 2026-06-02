package index

import (
	"bytes"
	"math/rand"
	"path/filepath"
	"testing"
)

func TestSerializeRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	b := NewBuilder(5000, 64, 5) // multi-cell: exercises centroids + CSR
	for i := 0; i < 5000; i++ {
		b.Add(randVec(rng), rng.Float64() < 0.44)
	}
	ix := b.Build()
	ix.SetNProbe(8)

	path := filepath.Join(t.TempDir(), "index.bin")
	if err := ix.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := LoadIndex(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got.SetNProbe(8) // nprobe is a runtime knob, not persisted; match the original



	if got.n != ix.n {
		t.Errorf("n = %d, want %d", got.n, ix.n)
	}
	if !bytes.Equal(u16ToBytes(got.data), u16ToBytes(ix.data)) {
		t.Error("data differs after round-trip")
	}
	if len(got.fraud) != len(ix.fraud) {
		t.Fatalf("fraud len = %d, want %d", len(got.fraud), len(ix.fraud))
	}
	for i := range ix.fraud {
		if got.fraud[i] != ix.fraud[i] {
			t.Fatalf("fraud[%d] differs", i)
		}
	}
	if got.bucketStart != ix.bucketStart {
		t.Fatalf("bucketStart = %v, want %v", got.bucketStart, ix.bucketStart)
	}
	if got.nlist != ix.nlist {
		t.Fatalf("nlist = %d, want %d", got.nlist, ix.nlist)
	}
	for i := range ix.cellStart {
		if got.cellStart[i] != ix.cellStart[i] {
			t.Fatalf("cellStart[%d] = %d, want %d", i, got.cellStart[i], ix.cellStart[i])
		}
	}
	for i := range ix.centroids {
		if got.centroids[i] != ix.centroids[i] {
			t.Fatalf("centroids[%d] = %v, want %v", i, got.centroids[i], ix.centroids[i])
		}
	}

	// Scores must be identical after a reload.
	for j := 0; j < 500; j++ {
		q := randVec(rng)
		if got.Score(q) != ix.Score(q) {
			t.Fatalf("score mismatch after reload for query %v", q)
		}
	}
}

func TestLoadRejectsGarbage(t *testing.T) {
	if _, err := readFrom(bytes.NewReader([]byte("not an index"))); err == nil {
		t.Error("expected error loading garbage, got nil")
	}
}

func u16ToBytes(s []uint16) []byte {
	b := make([]byte, len(s)*2)
	for i, v := range s {
		b[i*2] = byte(v)
		b[i*2+1] = byte(v >> 8)
	}
	return b
}
