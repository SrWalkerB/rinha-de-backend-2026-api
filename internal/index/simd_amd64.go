//go:build amd64

package index

import "os"

// SIMD distance kernels (hand-written Plan9 AVX2/FMA assembly in simd_amd64.s).
//
// Go has no SIMD intrinsics and does not auto-vectorize the float reduction in
// the distance loops, so the scalar dist2/centroidDist run 8-16x slower than the
// equivalent C/Rust with AVX2. These kernels recover that gap. They stay within
// the pure-Go toolchain (CGO disabled, no external deps): just an architecture-
// specific .s file plus a portable scalar fallback (simd_stub.go on non-amd64,
// and the scalar branch taken at runtime when the CPU lacks AVX2).
//
// useSIMD gates the dispatch. It is a var (not const) so tests can force the
// scalar path and assert the kernels match it bit-for-bit (within FP rounding).

//go:noescape
func cpuHasAVX2() bool

// centroidDistAVX2 returns Σ_{i<Dims} (q[i]-c[i])² for the 14-dim query q and
// centroid c (both float64, read in place, no allocation). q and c must each
// point to at least Dims contiguous float64s.
//
//go:noescape
func centroidDistAVX2(q, c *float64) float64

// dist2AVX2 returns Σ_{i<Dims} (q[i]-dequant(codes[i]))² where dequant(u) =
// (u-1)/10000 for u≥1 and -1 for u==0 (the null sentinel). q is the un-quantized
// float64 query (≥Dims elements); codes points to ≥Dims uint16 row codes. The
// dequant is folded inline (no dequantTab gather); u==0 is handled by a blend.
// Uses *1e-4 (vs scalar /10000), so it differs from scalar dist2 by ~1 ULP — used
// only to FILTER candidates; the exact 5-NN ranking comes from scalar dist2.
//
//go:noescape
func dist2AVX2(q *float64, codes *uint16) float64

// dist2BatchAVX2 computes out[r] = Σ (q[i]-dequant(codes[r*Dims+i]))² for r in
// [0,n), with q and the dequant constants kept in registers across the row loop.
// This is the form used on the hot path: one call per contiguous row run amortizes
// the asm-call overhead that made the per-row dist2AVX2 a net loss. out must have
// at least n elements; codes at least n*Dims; q at least Dims.
//
//go:noescape
func dist2BatchAVX2(q *float64, codes *uint16, n int, out *float64)

// distSoAi16AVX2 computes integer squared L2 distances over a SoA (dim-major)
// block: soa[d*strideRows+r] is row r's dim-d uint16 code. out[r] = Σ_d
// (int32(code)-qcode[d])² for r in [0, nBlocks*8). qcode is the query quantized to
// the grid (14 int32). Processes 8 rows per YMM block in parallel (real ILP, unlike
// the per-row AoS kernels). Approximate (query quantized) — a FILTER; exactness via
// scalar float64 refine. Requires nBlocks*8 <= strideRows.
//
//go:noescape
func distSoAi16AVX2(qcode *int32, soa *uint16, strideRows, nBlocks int, out *int32)

// distSoAf64AVX2 computes EXACT float64 squared distances over a SoA (dim-major)
// block: out[r] = Σ_d (q[d]-dequant(soa[d*strideRows+r]))² for r in [0, nBlocks*4).
// 4 rows per YMM block (4 lanes = 4 rows). Bit-comparable to scalar dist2 (same
// dequantTab values, ~1 ULP from FP summation reorder) → drop-in for the cell scan
// with topK/E unchanged. Caller does the <4 remainder and pads soa by ≥8 uint16.
//
//go:noescape
func distSoAf64AVX2(q *float64, soa *uint16, strideRows, nBlocks int, out *float64)

// useSIMD is true when the running CPU supports AVX2+FMA (checked once at init).
// The eval host (Haswell) always does; dev machines without it fall back to the
// scalar loops, so the binary never SIGILLs. Set INDEX_NO_SIMD=1 to force the
// scalar path (A/B measurement, or a safety switch on an unknown host).
var useSIMD = cpuHasAVX2() && os.Getenv("INDEX_NO_SIMD") != "1"
