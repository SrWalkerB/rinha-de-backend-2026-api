// Package dataset loads the labeled reference vectors from references.json(.gz)
// into an index.Index. It decodes the JSON array as a stream so memory stays
// bounded even though the uncompressed file is hundreds of megabytes.
package dataset

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"rinha-fraud/internal/index"
)

type record struct {
	Vector [index.Dims]float64 `json:"vector"`
	Label  string              `json:"label"`
}

// Load reads the reference file at path (gzip-decoded when it ends in ".gz")
// and returns a built, ready-to-query index. capacityHint pre-sizes the builder;
// nlist/iters configure the per-bucket k-means (≤0 → defaults).
func Load(path string, capacityHint, nlist, iters int) (*index.Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var r io.Reader = bufio.NewReaderSize(f, 1<<20)
	if strings.HasSuffix(path, ".gz") {
		gz, err := gzip.NewReader(r)
		if err != nil {
			return nil, fmt.Errorf("gzip %s: %w", path, err)
		}
		defer gz.Close()
		r = gz
	}

	b := index.NewBuilder(capacityHint, nlist, iters)
	dec := json.NewDecoder(r)

	// Opening '[' of the array.
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("read array start: %w", err)
	}
	for dec.More() {
		var rec record
		if err := dec.Decode(&rec); err != nil {
			return nil, fmt.Errorf("decode record %d: %w", b.Len(), err)
		}
		b.Add(rec.Vector, rec.Label == "fraud")
	}
	return b.Build(), nil
}
