// Package dataset loads the labeled reference vectors from references.json(.gz)
// into a knn.Index. It decodes the JSON array as a stream so memory stays bounded
// even though the uncompressed file is hundreds of megabytes.
package dataset

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"rinha-fraud/internal/knn"
	"rinha-fraud/internal/vectorize"
)

type record struct {
	Vector [vectorize.Dims]float64 `json:"vector"`
	Label  string                  `json:"label"`
}

// Load reads the reference file at path (gzip-decoded when it ends in ".gz")
// and returns a populated index. capacityHint pre-sizes the index.
func Load(path string, capacityHint int) (*knn.Index, error) {
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

	ix := knn.NewIndex(capacityHint)
	dec := json.NewDecoder(r)

	// Opening '[' of the array.
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("read array start: %w", err)
	}
	for dec.More() {
		var rec record
		if err := dec.Decode(&rec); err != nil {
			return nil, fmt.Errorf("decode record %d: %w", ix.Len(), err)
		}
		ix.Add(rec.Vector, rec.Label == "fraud")
	}
	return ix, nil
}
