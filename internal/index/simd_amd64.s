//go:build amd64

#include "textflag.h"

// func cpuHasAVX2() bool
// Returns true iff the CPU supports AVX (with OS XSAVE) and AVX2.
TEXT ·cpuHasAVX2(SB), NOSPLIT, $0-1
	// leaf 1: ECX bit27 (OSXSAVE) and bit28 (AVX)
	MOVL  $1, AX
	XORL  CX, CX
	CPUID
	ANDL  $0x18000000, CX
	CMPL  CX, $0x18000000
	JNE   nosimd
	// leaf 7, subleaf 0: EBX bit5 (AVX2)
	MOVL  $7, AX
	XORL  CX, CX
	CPUID
	ANDL  $0x20, BX
	JZ    nosimd
	MOVB  $1, ret+0(FP)
	RET
nosimd:
	MOVB  $0, ret+0(FP)
	RET

// func centroidDistAVX2(q, c *float64) float64
// Returns Σ_{i<14} (q[i]-c[i])². Reads exactly 14 float64 from each of q and c.
// 14 dims = 3 groups of 4 (AVX2, VFMADD231PD) + a 2-lane (XMM) tail; no padding,
// no out-of-bounds read (matches the stride-14 centroid/query layout).
TEXT ·centroidDistAVX2(SB), NOSPLIT, $0-24
	MOVQ   q+0(FP), AX
	MOVQ   c+8(FP), BX
	VXORPD Y3, Y3, Y3          // acc = 0

	VMOVUPD 0(AX), Y0          // dims 0-3
	VMOVUPD 0(BX), Y1
	VSUBPD  Y1, Y0, Y2         // Y2 = q - c
	VFMADD231PD Y2, Y2, Y3     // acc += Y2*Y2

	VMOVUPD 32(AX), Y0         // dims 4-7
	VMOVUPD 32(BX), Y1
	VSUBPD  Y1, Y0, Y2
	VFMADD231PD Y2, Y2, Y3

	VMOVUPD 64(AX), Y0         // dims 8-11
	VMOVUPD 64(BX), Y1
	VSUBPD  Y1, Y0, Y2
	VFMADD231PD Y2, Y2, Y3

	VMOVUPD 96(AX), X0         // dims 12-13 (2 lanes)
	VMOVUPD 96(BX), X1
	VSUBPD  X1, X0, X0
	VMULPD  X0, X0, X0         // X0 = [d12², d13²]

	VEXTRACTF128 $1, Y3, X4    // X4 = acc[2,3]
	VADDPD  X4, X3, X3         // X3 = acc[0]+acc[2], acc[1]+acc[3]
	VADDPD  X0, X3, X3         // += [d12², d13²]
	VHADDPD X3, X3, X3         // X3[0] = lane0 + lane1
	VMOVSD  X3, ret+16(FP)
	VZEROUPPER
	RET

// IEEE-754 bit patterns for the dequant fold (q is float64; ref is uint16 code):
//   dequant(u) = (u-1)/10000 for u>=1, and -1.0 for u==0 (null sentinel).
// We fold it inline: dq = (float64(u) - 1.0) * 1e-4, then blend -1.0 where u==0.
// (The *1e-4 vs /10000 differs by ~1 ULP, harmless: this kernel only FILTERS
//  candidates; the exact 5-NN comes from the scalar dist2 refine.)
DATA  fone<>+0(SB)/8,    $0x3FF0000000000000   // 1.0
GLOBL fone<>(SB),        RODATA|NOPTR, $8
DATA  fscale<>+0(SB)/8,  $0x3F1A36E2EB1C432D   // 1e-4
GLOBL fscale<>(SB),      RODATA|NOPTR, $8
DATA  fnegone<>+0(SB)/8, $0xBFF0000000000000   // -1.0
GLOBL fnegone<>(SB),     RODATA|NOPTR, $8

// dist2AVX2(q *float64, codes *uint16) float64
// Returns Σ_{i<14} (q[i]-dequant(codes[i]))². q: 14 float64; codes: 14 uint16.
// dims 0-11 in 3 AVX2 groups of 4 + a 2-lane XMM tail (no OOB past the 14 codes).
TEXT ·dist2AVX2(SB), NOSPLIT, $0-24
	MOVQ    q+0(FP), AX
	MOVQ    codes+8(FP), BX
	VXORPD  Y3, Y3, Y3            // acc = 0
	VXORPD  Y8, Y8, Y8            // zero (u==0 compare)
	VBROADCASTSD fone<>(SB), Y10
	VBROADCASTSD fscale<>(SB), Y12
	VBROADCASTSD fnegone<>(SB), Y13

	// --- dims 0-3 ---
	VPMOVZXWD (BX), X6           // 4 uint16 -> 4 uint32
	VCVTDQ2PD X6, Y7             // -> 4 float64 (uf)
	VCMPPD  $0, Y8, Y7, Y9       // mask = (uf == 0)
	VSUBPD  Y10, Y7, Y11         // uf - 1
	VMULPD  Y12, Y11, Y11        // (uf-1)*1e-4
	VANDPD  Y13, Y9, Y14         // -1.0 where u==0 else 0
	VANDNPD Y11, Y9, Y15         // dq    where u!=0 else 0
	VORPD   Y14, Y15, Y11        // dq blended
	VMOVUPD (AX), Y0             // q[0..3]
	VSUBPD  Y11, Y0, Y2          // q - dq
	VFMADD231PD Y2, Y2, Y3       // acc += diff²

	// --- dims 4-7 ---
	VPMOVZXWD 8(BX), X6
	VCVTDQ2PD X6, Y7
	VCMPPD  $0, Y8, Y7, Y9
	VSUBPD  Y10, Y7, Y11
	VMULPD  Y12, Y11, Y11
	VANDPD  Y13, Y9, Y14
	VANDNPD Y11, Y9, Y15
	VORPD   Y14, Y15, Y11
	VMOVUPD 32(AX), Y0
	VSUBPD  Y11, Y0, Y2
	VFMADD231PD Y2, Y2, Y3

	// --- dims 8-11 ---
	VPMOVZXWD 16(BX), X6
	VCVTDQ2PD X6, Y7
	VCMPPD  $0, Y8, Y7, Y9
	VSUBPD  Y10, Y7, Y11
	VMULPD  Y12, Y11, Y11
	VANDPD  Y13, Y9, Y14
	VANDNPD Y11, Y9, Y15
	VORPD   Y14, Y15, Y11
	VMOVUPD 64(AX), Y0
	VSUBPD  Y11, Y0, Y2
	VFMADD231PD Y2, Y2, Y3

	// --- dims 12-13 (2-lane tail; load only 4 bytes to avoid OOB) ---
	MOVL    24(BX), R8           // codes[12],codes[13] zero-extended
	MOVQ    R8, X5
	VPMOVZXWD X5, X6             // low 2 uint16 real, upper 2 are zero
	VCVTDQ2PD X6, X7             // 2 float64
	VCMPPD  $0, X8, X7, X9
	VSUBPD  X10, X7, X11
	VMULPD  X12, X11, X11
	VANDPD  X13, X9, X14
	VANDNPD X11, X9, X15
	VORPD   X14, X15, X11
	VMOVUPD 96(AX), X0           // q[12],q[13]
	VSUBPD  X11, X0, X2
	VMULPD  X2, X2, X2           // [d12², d13²]

	// horizontal sum: Y3 (4 lanes) + X2 (2 lanes)
	VEXTRACTF128 $1, Y3, X4
	VADDPD  X4, X3, X3
	VADDPD  X2, X3, X3
	VHADDPD X3, X3, X3
	VMOVSD  X3, ret+16(FP)
	VZEROUPPER
	RET

// dist2BatchAVX2(q *float64, codes *uint16, n int, out *float64)
// Computes out[r] = Σ_{i<14} (q[i]-dequant(codes[r*14+i]))² for r in [0,n).
// q and the 4 constants stay resident in YMM registers across the whole row loop,
// so the asm CALL + the q/const loads are paid ONCE per chunk, not per row (the
// per-row dist2AVX2 call was ~6x slower than the inlined scalar loop). Reads 14
// uint16 per row in 3×m64 groups + a 4-byte tail → no OOB past the row.
TEXT ·dist2BatchAVX2(SB), NOSPLIT, $0-32
	MOVQ    q+0(FP), AX
	MOVQ    codes+8(FP), BX
	MOVQ    n+16(FP), CX
	MOVQ    out+24(FP), DI
	CMPQ    CX, $0
	JLE     done
	// q resident: Y0=q0-3, Y1=q4-7, Y2=q8-11, X3=q12-13
	VMOVUPD 0(AX), Y0
	VMOVUPD 32(AX), Y1
	VMOVUPD 64(AX), Y2
	VMOVUPD 96(AX), X3
	// constants resident: Y4=0, Y5=1, Y6=1e-4, Y7=-1
	VXORPD  Y4, Y4, Y4
	VBROADCASTSD fone<>(SB), Y5
	VBROADCASTSD fscale<>(SB), Y6
	VBROADCASTSD fnegone<>(SB), Y7

rowloop:
	// --- dims 0-3 ---
	VPMOVZXWD (BX), X8
	VCVTDQ2PD X8, Y9
	VCMPPD  $0, Y4, Y9, Y10
	VSUBPD  Y5, Y9, Y11
	VMULPD  Y6, Y11, Y11
	VANDPD  Y7, Y10, Y12
	VANDNPD Y11, Y10, Y13
	VORPD   Y12, Y13, Y11
	VSUBPD  Y11, Y0, Y14
	VMULPD  Y14, Y14, Y15        // acc = diff²

	// --- dims 4-7 ---
	VPMOVZXWD 8(BX), X8
	VCVTDQ2PD X8, Y9
	VCMPPD  $0, Y4, Y9, Y10
	VSUBPD  Y5, Y9, Y11
	VMULPD  Y6, Y11, Y11
	VANDPD  Y7, Y10, Y12
	VANDNPD Y11, Y10, Y13
	VORPD   Y12, Y13, Y11
	VSUBPD  Y11, Y1, Y14
	VFMADD231PD Y14, Y14, Y15

	// --- dims 8-11 ---
	VPMOVZXWD 16(BX), X8
	VCVTDQ2PD X8, Y9
	VCMPPD  $0, Y4, Y9, Y10
	VSUBPD  Y5, Y9, Y11
	VMULPD  Y6, Y11, Y11
	VANDPD  Y7, Y10, Y12
	VANDNPD Y11, Y10, Y13
	VORPD   Y12, Y13, Y11
	VSUBPD  Y11, Y2, Y14
	VFMADD231PD Y14, Y14, Y15

	// --- dims 12-13 (2-lane tail; 4-byte load avoids OOB) ---
	MOVL    24(BX), R8
	MOVQ    R8, X8
	VPMOVZXWD X8, X8
	VCVTDQ2PD X8, X9
	VCMPPD  $0, X4, X9, X10
	VSUBPD  X5, X9, X11
	VMULPD  X6, X11, X11
	VANDPD  X7, X10, X12
	VANDNPD X11, X10, X13
	VORPD   X12, X13, X11
	VSUBPD  X11, X3, X14
	VMULPD  X14, X14, X14         // [d12², d13²]

	// horizontal sum: Y15 (4) + X14 (2)
	VEXTRACTF128 $1, Y15, X8
	VADDPD  X8, X15, X15
	VADDPD  X14, X15, X15
	VHADDPD X15, X15, X15
	VMOVSD  X15, (DI)

	ADDQ    $28, BX               // next row: 14 uint16
	ADDQ    $8, DI                // next out
	DECQ    CX
	JNZ     rowloop
	VZEROUPPER
done:
	RET

// distSoAi16AVX2(qcode *int32, soa *uint16, strideRows, nBlocks int, out *int32)
// SoA (dim-major) INTEGER distance, 8 rows per YMM lane-block. soa[d*strideRows+r]
// is row r's dim-d uint16 code. For each block of 8 rows: acc(8 int32) = Σ_d
// (code - qcode[d])². Integer squared L2 (q quantized to the grid) — a fast,
// sentinel-natural (quantize(-1)=0) FILTER; the exact float64 5-NN comes from a
// scalar refine. Processes nBlocks*8 rows; caller handles the <8 remainder.
// Requires nBlocks*8 <= strideRows (no OOB across dim segments).
TEXT ·distSoAi16AVX2(SB), NOSPLIT, $0-40
	MOVQ    qcode+0(FP), R13
	MOVQ    soa+8(FP), BX
	MOVQ    strideRows+16(FP), R9
	SHLQ    $1, R9                // strideBytes = strideRows*2
	MOVQ    nBlocks+24(FP), R10
	MOVQ    out+32(FP), R11
	XORQ    R12, R12              // byte offset of current block (b*16)
	CMPQ    R10, $0
	JLE     soadone
blockloop:
	MOVQ    BX, SI
	ADDQ    R12, SI               // SI -> soa[dim0, row b*8]
	MOVQ    R13, DI               // DI -> qcode[0]
	VPXOR   Y0, Y0, Y0            // acc (8 int32)
	MOVQ    $14, CX
dimloop:
	VPMOVZXWD (SI), Y1            // 8 uint16 codes -> 8 int32
	VPBROADCASTD (DI), Y2         // qcode[d]
	VPSUBD  Y2, Y1, Y1            // code - qcode[d]
	VPMULLD Y1, Y1, Y1           // diff²
	VPADDD  Y1, Y0, Y0           // acc += diff²
	ADDQ    R9, SI                // next dim segment
	ADDQ    $4, DI                // next qcode dim
	DECQ    CX
	JNZ     dimloop
	VMOVDQU Y0, (R11)            // store 8 distances
	ADDQ    $32, R11
	ADDQ    $16, R12
	DECQ    R10
	JNZ     blockloop
	VZEROUPPER
soadone:
	RET

// distSoAf64AVX2(q *float64, soa *uint16, strideRows, nBlocks int, out *float64)
// SoA (dim-major) EXACT float64 distance, 4 rows per YMM block (4 lanes = 4 rows).
// out[r] = Σ_d (q[d]-dequant(soa[d*strideRows+r]))² — bit-comparable to scalar dist2
// (same dequantTab values; differs only by FP summation reordering ~1 ULP). Used to
// drop-in replace the IVF cell scan with EXACT distances → topK/E unchanged.
// Processes nBlocks*4 rows; caller does the <4 remainder. Reads 4 uint16/dim (8B);
// caller pads the soa array by ≥8 uint16 so the last block's m64 read can't OOB.
TEXT ·distSoAf64AVX2(SB), NOSPLIT, $0-40
	MOVQ    q+0(FP), R13
	MOVQ    soa+8(FP), BX
	MOVQ    strideRows+16(FP), R9
	SHLQ    $1, R9                // strideBytes = strideRows*2
	MOVQ    nBlocks+24(FP), R10
	MOVQ    out+32(FP), R11
	XORQ    R12, R12              // byte offset of current block (b*8: 4 rows*2B)
	CMPQ    R10, $0
	JLE     f64done
	VXORPD  Y4, Y4, Y4           // zero (u==0 compare)
	VBROADCASTSD fone<>(SB), Y5
	VBROADCASTSD fscale<>(SB), Y6
	VBROADCASTSD fnegone<>(SB), Y7
f64block:
	MOVQ    BX, SI
	ADDQ    R12, SI               // SI -> soa[dim0, row b*4]
	MOVQ    R13, DI               // DI -> q[0]
	VXORPD  Y0, Y0, Y0           // acc (4 rows)
	MOVQ    $14, CX
f64dim:
	VPMOVZXWD (SI), X1            // 4 uint16 (4 rows, dim d) -> 4 int32
	VCVTDQ2PD X1, Y2             // -> 4 float64 (uf)
	VCMPPD  $0, Y4, Y2, Y3        // mask = (uf == 0)
	VSUBPD  Y5, Y2, Y8           // uf - 1
	VMULPD  Y6, Y8, Y8           // (uf-1)*1e-4
	VANDPD  Y7, Y3, Y9           // -1.0 where u==0
	VANDNPD Y8, Y3, Y10          // dq    where u!=0
	VORPD   Y9, Y10, Y8          // dq blended
	VBROADCASTSD (DI), Y11        // q[d]
	VSUBPD  Y8, Y11, Y12         // q[d] - dq
	VFMADD231PD Y12, Y12, Y0     // acc += diff²
	ADDQ    R9, SI                // next dim segment
	ADDQ    $8, DI                // next q dim
	DECQ    CX
	JNZ     f64dim
	VMOVUPD Y0, (R11)            // store 4 distances
	ADDQ    $32, R11
	ADDQ    $8, R12
	DECQ    R10
	JNZ     f64block
	VZEROUPPER
f64done:
	RET
