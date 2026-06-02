// Command buildindex constructs the partitioned search index offline — at
// image-build time, with the full machine (no CPU cap) — and writes it to a
// binary file. At startup the API just loads that file, so /ready is fast and
// the runtime never pays the build cost under the CPU cap.
package main

import (
	"log"
	"os"
	"strconv"
	"time"

	"rinha-fraud/internal/dataset"
	"rinha-fraud/internal/index"
)

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envi(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func main() {
	ref := env("REFERENCES_PATH", "./resources/references.json.gz")
	out := env("INDEX_OUT", "./resources/index.bin")
	capHint := envi("REFERENCES_CAPACITY", 3_000_000)
	nlist := envi("INDEX_NLIST", index.DefaultNList)
	iters := envi("INDEX_KMEANS_ITERS", index.DefaultKMeansIters)

	log.Printf("buildindex: loading + IVF-partitioning references from %s (nlist=%d iters=%d) ...", ref, nlist, iters)
	t0 := time.Now()
	ix, err := dataset.Load(ref, capHint, nlist, iters)
	if err != nil {
		log.Fatalf("load references: %v", err)
	}
	log.Printf("built index over %d vectors (nlist=%d) in %s", ix.Len(), ix.NList(), time.Since(t0))

	if err := ix.Save(out); err != nil {
		log.Fatalf("save index: %v", err)
	}
	if fi, err := os.Stat(out); err == nil {
		log.Printf("wrote %s (%.1f MB)", out, float64(fi.Size())/(1<<20))
	} else {
		log.Printf("wrote %s", out)
	}
}
