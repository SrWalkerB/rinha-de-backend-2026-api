//go:build !amd64

package index

// Non-amd64 fallback: no SIMD. useSIMD is a const false so the dispatch branches
// in index.go compile to the scalar path only (the AVX2 stubs are never called).

const useSIMD = false

func centroidDistAVX2(q, c *float64) float64                      { panic("centroidDistAVX2: no SIMD on this arch") }
func dist2AVX2(q *float64, codes *uint16) float64                 { panic("dist2AVX2: no SIMD on this arch") }
func dist2BatchAVX2(q *float64, codes *uint16, n int, out *float64) { panic("dist2BatchAVX2: no SIMD on this arch") }
func distSoAi16AVX2(qcode *int32, soa *uint16, strideRows, nBlocks int, out *int32) { panic("distSoAi16AVX2: no SIMD on this arch") }
func distSoAf64AVX2(q *float64, soa *uint16, strideRows, nBlocks int, out *float64) { panic("distSoAf64AVX2: no SIMD on this arch") }
