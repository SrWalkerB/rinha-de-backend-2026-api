//go:build amd64

package index

import (
	"math"
	"math/rand"
	"testing"
)

// scalarCentroidDist mirrors the scalar branch of centroidDist for the parity test.
func scalarCentroidDist(q, c *[Dims]float64) float64 {
	var d float64
	for i := 0; i < Dims; i++ {
		diff := q[i] - c[i]
		d += diff * diff
	}
	return d
}

func TestCentroidDistAVX2MatchesScalar(t *testing.T) {
	if !useSIMD {
		t.Skip("AVX2 unavailable on this CPU")
	}
	rng := rand.New(rand.NewSource(1))
	var maxRel float64
	for iter := 0; iter < 200000; iter++ {
		var q, c [Dims]float64
		for i := 0; i < Dims; i++ {
			q[i] = rng.Float64()*2 - 1 // [-1,1], covers the sentinel range
			c[i] = rng.Float64()*2 - 1
		}
		got := centroidDistAVX2(&q[0], &c[0])
		want := scalarCentroidDist(&q, &c)
		rel := math.Abs(got-want) / (1 + math.Abs(want))
		if rel > maxRel {
			maxRel = rel
		}
		if rel > 1e-12 {
			t.Fatalf("iter %d: simd=%.17g scalar=%.17g rel=%.3g\nq=%v\nc=%v", iter, got, want, rel, q, c)
		}
	}
	t.Logf("centroidDistAVX2 max relative error vs scalar over 200k: %.3g", maxRel)
}

// scalarDist2Codes mirrors the scalar branch of dist2 (uses dequantTab, exact).
func scalarDist2Codes(q *[Dims]float64, codes *[Dims]uint16) float64 {
	var d float64
	for i := 0; i < Dims; i++ {
		diff := q[i] - dequantTab[codes[i]]
		d += diff * diff
	}
	return d
}

func TestDist2AVX2MatchesScalar(t *testing.T) {
	if !useSIMD {
		t.Skip("AVX2 unavailable on this CPU")
	}
	rng := rand.New(rand.NewSource(7))
	var maxRel float64
	for iter := 0; iter < 300000; iter++ {
		var q [Dims]float64
		var codes [Dims]uint16
		for i := 0; i < Dims; i++ {
			q[i] = rng.Float64() // [0,1)
			// codes: mostly real [1,10001], with ~1/8 sentinel 0 to exercise the blend.
			if rng.Intn(8) == 0 {
				codes[i] = 0
				q[i] = -1 // matching sentinel query value
			} else {
				codes[i] = uint16(rng.Intn(maxBucketVal) + 1)
			}
		}
		got := dist2AVX2(&q[0], &codes[0])
		want := scalarDist2Codes(&q, &codes)
		rel := math.Abs(got-want) / (1 + math.Abs(want))
		if rel > maxRel {
			maxRel = rel
		}
		if rel > 1e-9 {
			t.Fatalf("iter %d: simd=%.17g scalar=%.17g rel=%.3g\nq=%v\ncodes=%v", iter, got, want, rel, q, codes)
		}
	}
	t.Logf("dist2AVX2 max relative error vs scalar over 300k (incl. sentinels): %.3g", maxRel)
}

func TestDist2BatchAVX2MatchesScalar(t *testing.T) {
	if !useSIMD {
		t.Skip("AVX2 unavailable on this CPU")
	}
	rng := rand.New(rand.NewSource(11))
	const n = 777
	codes := make([]uint16, n*Dims)
	for i := range codes {
		if rng.Intn(8) == 0 {
			codes[i] = 0
		} else {
			codes[i] = uint16(rng.Intn(maxBucketVal) + 1)
		}
	}
	var q [Dims]float64
	for i := 0; i < Dims; i++ {
		q[i] = rng.Float64()*2 - 1
	}
	out := make([]float64, n)
	dist2BatchAVX2(&q[0], &codes[0], n, &out[0])
	var maxRel float64
	for r := 0; r < n; r++ {
		var cs [Dims]uint16
		copy(cs[:], codes[r*Dims:])
		want := scalarDist2Codes(&q, &cs)
		rel := math.Abs(out[r]-want) / (1 + math.Abs(want))
		if rel > maxRel {
			maxRel = rel
		}
		if rel > 1e-9 {
			t.Fatalf("row %d: batch=%.17g scalar=%.17g rel=%.3g", r, out[r], want, rel)
		}
	}
	t.Logf("dist2BatchAVX2 max relative error over %d rows: %.3g", n, maxRel)
}

func TestDist2AVX2AllSentinel(t *testing.T) {
	if !useSIMD {
		t.Skip("AVX2 unavailable on this CPU")
	}
	// Null bucket: every code 0 (dequant -1), query -1 → exact distance 0.
	var q [Dims]float64
	var codes [Dims]uint16
	for i := 0; i < Dims; i++ {
		q[i] = -1
		codes[i] = 0
	}
	if got := dist2AVX2(&q[0], &codes[0]); math.Abs(got) > 1e-12 {
		t.Fatalf("all-sentinel dist2AVX2 = %v, want 0", got)
	}
}

// BenchmarkRowScan measures a contiguous row scan (like a cell/bucket scan):
// scalar (inlined) vs SIMD per-row call (includes asm call overhead). This is the
// realistic kernel A/B — the real win for brute needs amortizing the call.
func BenchmarkRowScan(b *testing.B) {
	const rows = 4096
	data := make([]uint16, rows*Dims)
	rng := rand.New(rand.NewSource(3))
	for i := range data {
		data[i] = uint16(rng.Intn(maxBucketVal) + 1)
	}
	var q [Dims]float64
	for i := 0; i < Dims; i++ {
		q[i] = rng.Float64()
	}
	b.Run("scalar", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			var acc float64
			for r := 0; r < rows; r++ {
				off := r * Dims
				var d float64
				for j := 0; j < Dims; j++ {
					diff := q[j] - dequantTab[data[off+j]]
					d += diff * diff
				}
				acc += d
			}
			sink = acc
		}
	})
	b.Run("simd_percall", func(b *testing.B) {
		if !useSIMD {
			b.Skip("no AVX2")
		}
		for n := 0; n < b.N; n++ {
			var acc float64
			for r := 0; r < rows; r++ {
				acc += dist2AVX2(&q[0], &data[r*Dims])
			}
			sink = acc
		}
	})
	b.Run("simd_batch", func(b *testing.B) {
		if !useSIMD {
			b.Skip("no AVX2")
		}
		out := make([]float64, rows)
		for n := 0; n < b.N; n++ {
			dist2BatchAVX2(&q[0], &data[0], rows, &out[0])
			var acc float64
			for r := 0; r < rows; r++ {
				acc += out[r]
			}
			sink = acc
		}
	})
}

var sink float64

// buildSoA lays out n rows of Dims uint16 codes dim-major: soa[d*n+r].
func buildSoA(n int, rng *rand.Rand) (soa []uint16, qcode []int32) {
	soa = make([]uint16, Dims*n)
	for d := 0; d < Dims; d++ {
		for r := 0; r < n; r++ {
			soa[d*n+r] = uint16(rng.Intn(maxBucketVal) + 1)
		}
	}
	qcode = make([]int32, Dims)
	for d := 0; d < Dims; d++ {
		qcode[d] = int32(rng.Intn(maxBucketVal) + 1)
	}
	return
}

func TestDistSoAi16MatchesScalar(t *testing.T) {
	if !useSIMD {
		t.Skip("AVX2 unavailable on this CPU")
	}
	rng := rand.New(rand.NewSource(13))
	const n = 8 * 100 // multiple of 8
	soa, qcode := buildSoA(n, rng)
	out := make([]int32, n)
	distSoAi16AVX2(&qcode[0], &soa[0], n, n/8, &out[0])
	for r := 0; r < n; r++ {
		var want int32
		for d := 0; d < Dims; d++ {
			diff := int32(soa[d*n+r]) - qcode[d]
			want += diff * diff
		}
		if out[r] != want {
			t.Fatalf("row %d: simd=%d scalar=%d", r, out[r], want)
		}
	}
}

// BenchmarkSoAScan: SoA int16 8-wide kernel over 4096 rows vs the scalar AoS
// float64 loop (BenchmarkRowScan/scalar) — same row count, comparable ns/op.
func BenchmarkSoAScan(b *testing.B) {
	if !useSIMD {
		b.Skip("no AVX2")
	}
	const rows = 4096
	rng := rand.New(rand.NewSource(3))
	soa, qcode := buildSoA(rows, rng)
	out := make([]int32, rows)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		distSoAi16AVX2(&qcode[0], &soa[0], rows, rows/8, &out[0])
		var acc int64
		for r := 0; r < rows; r++ {
			acc += int64(out[r])
		}
		sink = float64(acc)
	}
}

func TestDistSoAf64MatchesScalar(t *testing.T) {
	if !useSIMD {
		t.Skip("AVX2 unavailable on this CPU")
	}
	rng := rand.New(rand.NewSource(17))
	const n = 4 * 137 // multiple of 4
	soa := make([]uint16, Dims*n+8) // +8 pad for the last-block m64 read
	for d := 0; d < Dims; d++ {
		for r := 0; r < n; r++ {
			if rng.Intn(8) == 0 {
				soa[d*n+r] = 0 // sentinel
			} else {
				soa[d*n+r] = uint16(rng.Intn(maxBucketVal) + 1)
			}
		}
	}
	var q [Dims]float64
	for d := 0; d < Dims; d++ {
		q[d] = rng.Float64()*2 - 1
	}
	out := make([]float64, n)
	distSoAf64AVX2(&q[0], &soa[0], n, n/4, &out[0])
	var maxRel float64
	for r := 0; r < n; r++ {
		var want float64
		for d := 0; d < Dims; d++ {
			diff := q[d] - dequantTab[soa[d*n+r]]
			want += diff * diff
		}
		rel := math.Abs(out[r]-want) / (1 + math.Abs(want))
		if rel > maxRel {
			maxRel = rel
		}
		if rel > 1e-9 {
			t.Fatalf("row %d: simd=%.17g scalar=%.17g rel=%.3g", r, out[r], want, rel)
		}
	}
	t.Logf("distSoAf64AVX2 max relative error over %d rows: %.3g", n, maxRel)
}

func BenchmarkSoAScanF64(b *testing.B) {
	if !useSIMD {
		b.Skip("no AVX2")
	}
	const rows = 4096
	rng := rand.New(rand.NewSource(3))
	soa := make([]uint16, Dims*rows+8)
	for i := 0; i < Dims*rows; i++ {
		soa[i] = uint16(rng.Intn(maxBucketVal) + 1)
	}
	var q [Dims]float64
	for d := 0; d < Dims; d++ {
		q[d] = rng.Float64()
	}
	out := make([]float64, rows)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		distSoAf64AVX2(&q[0], &soa[0], rows, rows/4, &out[0])
		var acc float64
		for r := 0; r < rows; r++ {
			acc += out[r]
		}
		sink = acc
	}
}

func TestCentroidDistAVX2ZeroAlloc(t *testing.T) {
	if !useSIMD {
		t.Skip("AVX2 unavailable on this CPU")
	}
	var q, c [Dims]float64
	for i := 0; i < Dims; i++ {
		q[i] = float64(i) * 0.07
		c[i] = float64(i) * 0.05
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_ = centroidDistAVX2(&q[0], &c[0])
	})
	if allocs != 0 {
		t.Fatalf("centroidDistAVX2 allocates %v/op, want 0", allocs)
	}
}
