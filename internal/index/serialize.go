package index

// Binary (de)serialization of a built Index. The IVF structure is built once at
// image-build time (full CPU) and loaded quickly at startup, so /ready is fast.
//
// Layout (little-endian):
//
//	header : magic u32 ("RFI2") | version u32 | dims u32 | k u32
//	body   : n u64 | nlist u32 | data(u16) | fraud(u64)
//	         | bucketStart(i32, len 17) | centroids(f64) | cellStart(i32)
//
// Length-prefixed slices keep the format self-describing.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	indexMagic   uint32 = 0x32494652 // "RFI2"
	indexVersion uint32 = 4          // v4: SoA (dim-major per bucket) data + tail pad, for the int16 SIMD kernel
)

// Save writes the built index to path.
func (ix *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 1<<20)
	if err := ix.writeTo(w); err != nil {
		return err
	}
	return w.Flush()
}

// LoadIndex reads an index previously written by Save.
func LoadIndex(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return readFrom(bufio.NewReaderSize(f, 1<<20))
}

func (ix *Index) writeTo(w io.Writer) error {
	var hdr [16]byte
	binary.LittleEndian.PutUint32(hdr[0:], indexMagic)
	binary.LittleEndian.PutUint32(hdr[4:], indexVersion)
	binary.LittleEndian.PutUint32(hdr[8:], uint32(Dims))
	binary.LittleEndian.PutUint32(hdr[12:], uint32(K))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if err := writeU64(w, uint64(ix.n)); err != nil {
		return err
	}
	if err := writeU32(w, uint32(ix.nlist)); err != nil {
		return err
	}
	if err := writeU16Slice(w, ix.data); err != nil {
		return err
	}
	if err := writeU64Slice(w, ix.fraud); err != nil {
		return err
	}
	if err := writeI32Slice(w, ix.bucketStart[:]); err != nil {
		return err
	}
	if err := writeF64Slice(w, ix.centroids); err != nil {
		return err
	}
	return writeI32Slice(w, ix.cellStart)
}

func readFrom(r io.Reader) (*Index, error) {
	var hdr [16]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, fmt.Errorf("index header: %w", err)
	}
	if magic := binary.LittleEndian.Uint32(hdr[0:]); magic != indexMagic {
		return nil, fmt.Errorf("index: bad magic %#x", magic)
	}
	if v := binary.LittleEndian.Uint32(hdr[4:]); v != indexVersion {
		return nil, fmt.Errorf("index: unsupported version %d (want %d)", v, indexVersion)
	}
	if dims := binary.LittleEndian.Uint32(hdr[8:]); int(dims) != Dims {
		return nil, fmt.Errorf("index: dims %d != %d", dims, Dims)
	}
	if k := binary.LittleEndian.Uint32(hdr[12:]); int(k) != K {
		return nil, fmt.Errorf("index: k %d != %d", k, K)
	}

	n, err := readU64(r)
	if err != nil {
		return nil, err
	}
	nlist, err := readU32(r)
	if err != nil {
		return nil, err
	}
	data, err := readU16Slice(r)
	if err != nil {
		return nil, err
	}
	fraud, err := readU64Slice(r)
	if err != nil {
		return nil, err
	}
	bucketStart, err := readI32Slice(r)
	if err != nil {
		return nil, err
	}
	centroids, err := readF64Slice(r)
	if err != nil {
		return nil, err
	}
	cellStart, err := readI32Slice(r)
	if err != nil {
		return nil, err
	}
	if len(bucketStart) != numBuckets+1 {
		return nil, fmt.Errorf("index: bucketStart len %d != %d", len(bucketStart), numBuckets+1)
	}
	if len(data) != int(n)*Dims+simdTailPad {
		return nil, fmt.Errorf("index: data len %d != n*Dims+pad %d", len(data), int(n)*Dims+simdTailPad)
	}
	if len(cellStart) != numBuckets*int(nlist)+1 {
		return nil, fmt.Errorf("index: cellStart len %d != %d", len(cellStart), numBuckets*int(nlist)+1)
	}
	if len(centroids) != numBuckets*int(nlist)*Dims {
		return nil, fmt.Errorf("index: centroids len %d != %d", len(centroids), numBuckets*int(nlist)*Dims)
	}

	ix := &Index{
		data:      data,
		fraud:     fraud,
		n:         int(n),
		nlist:     int(nlist),
		centroids: centroids,
		cellStart: cellStart,
		nprobe:    int(nlist),
	}
	copy(ix.bucketStart[:], bucketStart)
	// centroidsI16 is derived, not serialized — build it from the loaded centroids so
	// the int16 cell-selection scan works on a loaded index (no format bump).
	ix.buildCentroidI16()
	return ix, nil
}

// --- low-level encoders ---

func writeU32(w io.Writer, v uint32) error {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	_, err := w.Write(b[:])
	return err
}

func writeU64(w io.Writer, v uint64) error {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	_, err := w.Write(b[:])
	return err
}

func writeU16Slice(w io.Writer, s []uint16) error {
	if err := writeU64(w, uint64(len(s))); err != nil {
		return err
	}
	buf := make([]byte, len(s)*2)
	for i, v := range s {
		binary.LittleEndian.PutUint16(buf[i*2:], v)
	}
	_, err := w.Write(buf)
	return err
}

func writeI32Slice(w io.Writer, s []int32) error {
	if err := writeU64(w, uint64(len(s))); err != nil {
		return err
	}
	buf := make([]byte, len(s)*4)
	for i, v := range s {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(v))
	}
	_, err := w.Write(buf)
	return err
}

func writeU64Slice(w io.Writer, s []uint64) error {
	if err := writeU64(w, uint64(len(s))); err != nil {
		return err
	}
	buf := make([]byte, len(s)*8)
	for i, v := range s {
		binary.LittleEndian.PutUint64(buf[i*8:], v)
	}
	_, err := w.Write(buf)
	return err
}

func writeF64Slice(w io.Writer, s []float64) error {
	if err := writeU64(w, uint64(len(s))); err != nil {
		return err
	}
	buf := make([]byte, len(s)*8)
	for i, v := range s {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	_, err := w.Write(buf)
	return err
}

// --- low-level decoders ---

func readU32(r io.Reader) (uint32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func readU64(r io.Reader) (uint64, error) {
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func readU16Slice(r io.Reader) ([]uint16, error) {
	n, err := readU64(r)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, int(n)*2)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	s := make([]uint16, int(n))
	for i := range s {
		s[i] = binary.LittleEndian.Uint16(buf[i*2:])
	}
	return s, nil
}

func readI32Slice(r io.Reader) ([]int32, error) {
	n, err := readU64(r)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, int(n)*4)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	s := make([]int32, int(n))
	for i := range s {
		s[i] = int32(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return s, nil
}

func readU64Slice(r io.Reader) ([]uint64, error) {
	n, err := readU64(r)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, int(n)*8)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	s := make([]uint64, int(n))
	for i := range s {
		s[i] = binary.LittleEndian.Uint64(buf[i*8:])
	}
	return s, nil
}

func readF64Slice(r io.Reader) ([]float64, error) {
	n, err := readU64(r)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, int(n)*8)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	s := make([]float64, int(n))
	for i := range s {
		s[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:]))
	}
	return s, nil
}
