package knn

// Binary (de)serialization of a built index. This lets the IVF structure be
// constructed once at image-build time (full CPU, large nlist) and loaded
// quickly at startup, instead of running k-means under the runtime CPU cap.
// See docs/performance/05-preprocessamento-no-build.md.
//
// Layout (little-endian):
//
//	header  : magic u32 | version u32 | dims u32 | mode u32
//	body    : n u64 | data(bytes) | bits(u64 slice)
//	if IVF  : nlist u32 | nprobe u32 | centroids(bytes) | counts(u32 slice) | ids(i32 slice)
//
// Length-prefixed slices keep the format self-describing and robust.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"rinha-fraud/internal/vectorize"
)

const (
	indexMagic   uint32 = 0x52464958 // "RFIX"
	indexVersion uint32 = 2          // v2: data/centroids are uint16 (was uint8 in v1)
)

// Save writes the built index to path. Only IVF and brute modes are
// serializable (the VP-tree is teaching-only and not used in production).
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
	binary.LittleEndian.PutUint32(hdr[8:], uint32(vectorize.Dims))
	binary.LittleEndian.PutUint32(hdr[12:], uint32(ix.mode))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if err := writeU64(w, uint64(ix.n)); err != nil {
		return err
	}
	if err := writeU16Slice(w, ix.data); err != nil {
		return err
	}
	if err := writeU64Slice(w, ix.bits); err != nil {
		return err
	}
	if ix.mode == modeIVF {
		if err := ix.ivf.writeTo(w); err != nil {
			return err
		}
	}
	return nil
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
	if dims := binary.LittleEndian.Uint32(hdr[8:]); int(dims) != vectorize.Dims {
		return nil, fmt.Errorf("index: dims %d != %d", dims, vectorize.Dims)
	}
	mode := searchMode(binary.LittleEndian.Uint32(hdr[12:]))

	n, err := readU64(r)
	if err != nil {
		return nil, err
	}
	data, err := readU16Slice(r)
	if err != nil {
		return nil, err
	}
	bits, err := readU64Slice(r)
	if err != nil {
		return nil, err
	}

	ix := &Index{data: data, bits: bits, n: int(n), mode: mode}
	if mode == modeIVF {
		f, err := readIVF(r)
		if err != nil {
			return nil, err
		}
		ix.ivf = f
	}
	return ix, nil
}

func (f *ivfIndex) writeTo(w io.Writer) error {
	if err := writeU32(w, uint32(f.nlist)); err != nil {
		return err
	}
	if err := writeU32(w, uint32(f.nprobe)); err != nil {
		return err
	}
	if err := writeU16Slice(w, f.centroids); err != nil {
		return err
	}

	// Per-cell counts, then all ids flattened in cell order. The reader rebuilds
	// the [][]int32 as sub-slices of one backing array (fewer allocations).
	counts := make([]uint32, f.nlist)
	total := 0
	for c := range f.lists {
		counts[c] = uint32(len(f.lists[c]))
		total += len(f.lists[c])
	}
	if err := writeU32Slice(w, counts); err != nil {
		return err
	}
	flat := make([]int32, 0, total)
	for c := range f.lists {
		flat = append(flat, f.lists[c]...)
	}
	return writeI32Slice(w, flat)
}

func readIVF(r io.Reader) (*ivfIndex, error) {
	nlist, err := readU32(r)
	if err != nil {
		return nil, err
	}
	nprobe, err := readU32(r)
	if err != nil {
		return nil, err
	}
	centroids, err := readU16Slice(r)
	if err != nil {
		return nil, err
	}
	counts, err := readU32Slice(r)
	if err != nil {
		return nil, err
	}
	flat, err := readI32Slice(r)
	if err != nil {
		return nil, err
	}
	if len(counts) != int(nlist) {
		return nil, fmt.Errorf("index: counts %d != nlist %d", len(counts), nlist)
	}

	f := &ivfIndex{
		centroids: centroids,
		lists:     make([][]int32, nlist),
		nlist:     int(nlist),
		nprobe:    int(nprobe),
	}
	off := 0
	for c := 0; c < int(nlist); c++ {
		cnt := int(counts[c])
		if off+cnt > len(flat) {
			return nil, fmt.Errorf("index: ids overflow at cell %d", c)
		}
		f.lists[c] = flat[off : off+cnt : off+cnt]
		off += cnt
	}
	return f, nil
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

func writeU32Slice(w io.Writer, s []uint32) error {
	if err := writeU64(w, uint64(len(s))); err != nil {
		return err
	}
	buf := make([]byte, len(s)*4)
	for i, v := range s {
		binary.LittleEndian.PutUint32(buf[i*4:], v)
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

func readU32Slice(r io.Reader) ([]uint32, error) {
	n, err := readU64(r)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, int(n)*4)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	s := make([]uint32, int(n))
	for i := range s {
		s[i] = binary.LittleEndian.Uint32(buf[i*4:])
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
