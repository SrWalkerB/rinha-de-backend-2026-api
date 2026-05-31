package knn

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestIVFRoundTrip: an IVF index saved and reloaded must be byte-identical in
// its stored vectors and produce the SAME score for every query — the baked
// index has to behave exactly like the one built at startup.
func TestIVFRoundTrip(t *testing.T) {
	const n = 6000
	orig := NewIndex(n)
	fillClustered(orig, n, 48, 3)
	orig.Build(BuildConfig{Mode: "ivf", NList: 96, NProbe: 8, Iters: 6})
	if orig.mode != modeIVF {
		t.Fatalf("setup: want ivf mode, got %d", orig.mode)
	}

	path := filepath.Join(t.TempDir(), "index.bin")
	if err := orig.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := LoadIndex(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if got.n != orig.n {
		t.Fatalf("n %d != %d", got.n, orig.n)
	}
	if got.mode != orig.mode {
		t.Fatalf("mode %d != %d", got.mode, orig.mode)
	}
	if !slices.Equal(got.data, orig.data) {
		t.Fatalf("data differs after round-trip")
	}
	if got.ivf.nlist != orig.ivf.nlist || got.ivf.nprobe != orig.ivf.nprobe {
		t.Fatalf("ivf params differ: got (%d,%d) want (%d,%d)",
			got.ivf.nlist, got.ivf.nprobe, orig.ivf.nlist, orig.ivf.nprobe)
	}
	if !slices.Equal(got.ivf.centroids, orig.ivf.centroids) {
		t.Fatalf("centroids differ after round-trip")
	}

	for i, q := range randomQueries(400, 77) {
		if a, b := got.Score(q), orig.Score(q); a != b {
			t.Fatalf("query %d: loaded score %v != original %v", i, a, b)
		}
	}
}

// TestLoadIndexRejectsGarbage: a corrupt/foreign file must error, not panic,
// so the startup fallback (build from references) can kick in.
func TestLoadIndexRejectsGarbage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	if err := os.WriteFile(path, []byte("not an index"), 0o600); err != nil {
		t.Fatalf("setup: %v", err)
	}
	if _, err := LoadIndex(path); err == nil {
		t.Fatalf("expected error loading garbage, got nil")
	}
}

// TestSetNProbe: retuning a loaded index changes how many cells are scanned and
// is clamped to the valid range.
func TestSetNProbe(t *testing.T) {
	const n = 6000
	ix := NewIndex(n)
	fillClustered(ix, n, 48, 9)
	ix.Build(BuildConfig{Mode: "ivf", NList: 96, NProbe: 4, Iters: 4})

	ix.SetNProbe(12)
	if ix.ivf.nprobe != 12 {
		t.Fatalf("nprobe = %d, want 12", ix.ivf.nprobe)
	}
	ix.SetNProbe(0) // no-op
	if ix.ivf.nprobe != 12 {
		t.Fatalf("nprobe = %d after no-op, want 12", ix.ivf.nprobe)
	}
	ix.SetNProbe(1 << 30) // clamp to maxProbe (or nlist)
	if ix.ivf.nprobe > maxProbe || ix.ivf.nprobe > ix.ivf.nlist {
		t.Fatalf("nprobe = %d, not clamped", ix.ivf.nprobe)
	}
}
