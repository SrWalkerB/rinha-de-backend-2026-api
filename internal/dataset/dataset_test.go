package dataset

import (
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"rinha-fraud/internal/vectorize"
)

// writeJSON renders records as the references.json array format.
func recordsJSON(vectors [][vectorize.Dims]float64, labels []string) []byte {
	var b bytes.Buffer
	b.WriteByte('[')
	for i := range vectors {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`{"vector":[`)
		for d, v := range vectors[i] {
			if d > 0 {
				b.WriteByte(',')
			}
			b.WriteString(strconv.FormatFloat(v, 'f', -1, 64))
		}
		b.WriteString(`],"label":"`)
		b.WriteString(labels[i])
		b.WriteString(`"}`)
	}
	b.WriteByte(']')
	return b.Bytes()
}

func sameVec(base float64) [vectorize.Dims]float64 {
	var v [vectorize.Dims]float64
	for i := range v {
		v[i] = base
	}
	return v
}

func TestLoadPlainJSON(t *testing.T) {
	vecs := [][vectorize.Dims]float64{
		sameVec(0.5), sameVec(0.5), sameVec(0.5), sameVec(0.5), sameVec(0.5),
	}
	labels := []string{"fraud", "fraud", "fraud", "fraud", "fraud"}
	path := filepath.Join(t.TempDir(), "refs.json")
	if err := os.WriteFile(path, recordsJSON(vecs, labels), 0o644); err != nil {
		t.Fatal(err)
	}

	ix, err := Load(path, 8, 0, 0)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if ix.Len() != 5 {
		t.Fatalf("Len = %d, want 5", ix.Len())
	}
	if s := ix.Score(sameVec(0.5)); s != 1.0 {
		t.Errorf("all-fraud Score = %v, want 1.0", s)
	}
}

func TestLoadGzip(t *testing.T) {
	vecs := [][vectorize.Dims]float64{
		sameVec(0.2), sameVec(0.2), sameVec(0.2), sameVec(0.2), sameVec(0.2),
	}
	labels := []string{"legit", "legit", "legit", "legit", "legit"}
	raw := recordsJSON(vecs, labels)

	var gzbuf bytes.Buffer
	gw := gzip.NewWriter(&gzbuf)
	if _, err := gw.Write(raw); err != nil {
		t.Fatal(err)
	}
	gw.Close()

	path := filepath.Join(t.TempDir(), "refs.json.gz")
	if err := os.WriteFile(path, gzbuf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	ix, err := Load(path, 8, 0, 0)
	if err != nil {
		t.Fatalf("Load gz: %v", err)
	}
	if ix.Len() != 5 {
		t.Fatalf("Len = %d, want 5", ix.Len())
	}
	if s := ix.Score(sameVec(0.2)); s != 0.0 {
		t.Errorf("all-legit Score = %v, want 0.0", s)
	}
}

// The real sample file from the challenge must load with its documented format.
func TestLoadRealExampleReferences(t *testing.T) {
	path := filepath.Join("..", "..", "resources", "example-references.json")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("sample not present: %v", err)
	}
	ix, err := Load(path, 128, 0, 0)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if ix.Len() != 100 {
		t.Errorf("Len = %d, want 100", ix.Len())
	}
}
