# Caminho A — Quantização lossless + distância float64 (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer a busca 5-NN reproduzir o gabarito `float64-exato` (0 failures no brute) guardando as referências em uint16 na grade nativa ×10000 (lossless, 84MB) e computando a distância em **float64** contra a query crua (não quantizada).

**Architecture:** Hoje `knn` quantiza ref E query pra uint16 (×65534) e calcula distância inteira (uint64). A query quantizada (~8e-6/dim) flipa ~144 empates de fronteira (doc 07). Fix: refs em uint16 via `round(v*10000)+1` (lossless, pois o dado é 4-casas), distância em float64 dequantizando a ref inline contra a query float64. Isso é algebricamente o `bruteF64` do `cmd/diag` (medido 0 failures). Centróides do IVF passam a `[]float64`. p99 ~igual (IVF varre poucas linhas; mul float64 ≈ int em volume). KNN melhorado é selecionado por `SCORER=knn` (Caminho C adiciona `model`); a imagem 1843 não é tocada até medirmos e escolhermos publicar.

**Tech Stack:** Go (stdlib, sem deps), `internal/knn` (`knn.go`/`ivf.go`/`vptree.go`/`serialize.go`), `cmd/buildindex`, `cmd/diag`, k6/docker (`run-test.ps1`).

**Spec:** `docs/superpowers/specs/2026-05-31-api-gbdt-e-quantizacao-design.md` (Caminho A).

**Pré-requisito de ambiente (toda sessão):** `$env:Path = "C:\Program Files\Go\bin;" + $env:Path`; working dir = `C:\Me\me-projects\rinha-backend-2026\rinha-de-backend-2026-api`.

---

### Task 0: Branch + baseline verde

**Files:** nenhum (setup).

- [ ] **Step 1: Garantir Go no PATH e dir certo**

Run (PowerShell):
```powershell
$env:Path = "C:\Program Files\Go\bin;" + $env:Path
Set-Location C:\Me\me-projects\rinha-backend-2026\rinha-de-backend-2026-api
go version
```
Expected: imprime `go version go1.x...`.

- [ ] **Step 2: Verificar baseline de testes verde ANTES de mexer**

Run: `go test ./...`
Expected: tudo `ok` (estado uint16 atual). Se algo já falha, parar e reportar — não começar refactor sobre base vermelha.

- [ ] **Step 3: Criar branch de feature (não trabalhar na default)**

Run:
```powershell
git rev-parse --abbrev-ref HEAD
git checkout -b feat/quantize-lossless-f64
```
Expected: `Switched to a new branch 'feat/quantize-lossless-f64'`.

---

### Task 1: `quantize` ×10000 + helper `dequant` (puro, isolado, ainda compila)

**Files:**
- Modify: `internal/knn/knn.go` (`quantize` ~40-48; adicionar `dequant`)
- Test: `internal/knn/knn_test.go` (`TestQuantizeMapping` ~18-32; adicionar `TestDequantRoundTrip`)

- [ ] **Step 1: Atualizar o teste de `quantize` e adicionar round-trip (RED)**

Em `internal/knn/knn_test.go`, substituir o `TestQuantizeMapping` (linhas 18-32) por:

```go
// quantize pins the uint16 storage mapping on the data's native 4-decimal grid:
// -1 sentinel -> 0, [0,1] -> [1,10001], monotonic.
func TestQuantizeMapping(t *testing.T) {
	if got := quantize(-1); got != 0 {
		t.Errorf("quantize(-1) = %d, want 0", got)
	}
	if got := quantize(0); got != 1 {
		t.Errorf("quantize(0) = %d, want 1", got)
	}
	if got := quantize(1); got != 10001 {
		t.Errorf("quantize(1) = %d, want 10001", got)
	}
	if quantize(0.5) <= quantize(0.25) {
		t.Errorf("quantize not monotonic: q(0.5)=%d q(0.25)=%d", quantize(0.5), quantize(0.25))
	}
}

// dequant must reverse quantize bit-exactly for 4-decimal values, so float64
// distance against an un-quantized query equals the ground-truth exact search.
func TestDequantRoundTrip(t *testing.T) {
	if got := dequant(0); got != -1 {
		t.Errorf("dequant(0) = %v, want -1 (sentinel)", got)
	}
	for _, v := range []float64{0, 0.0001, 0.0833, 0.3913, 0.8261, 0.9999, 1} {
		if got := dequant(quantize(v)); got != v {
			t.Errorf("dequant(quantize(%v)) = %v, want exact", v, got)
		}
	}
}
```

- [ ] **Step 2: Rodar — deve FALHAR (compila, asserts errados)**

Run: `go test ./internal/knn/ -run 'TestQuantizeMapping|TestDequantRoundTrip' -v`
Expected: FAIL — `quantize(1) = 65535, want 10001` e `dequant` indefinido/errado.

- [ ] **Step 3: Implementar `quantize` ×10000 + `dequant`**

Em `internal/knn/knn.go`, substituir o doc-comment + função `quantize` (linhas 31-48) por:

```go
// quantize maps a normalized dimension to a uint16 bucket on the data's native
// grid. The reference vectors are exactly 4-decimal, so round(v*10000) stores
// them losslessly; dequant then reproduces the parsed float64 bit-for-bit.
//
//	-1 (sentinel, no last_transaction) -> 0
//	[0, 1]                             -> [1, 10001]
//
// Distance is computed in float64 (see dist2) against the un-quantized query,
// so an exact (brute) search reproduces the float64 ground-truth 5-NN with no
// quantization flips (the ~144 flips under the old ×65534 scale came from
// quantizing the full-precision query; see docs/performance/07 and /08).
func quantize(v float64) uint16 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		v = 1
	}
	return uint16(math.Round(v*10000)) + 1
}

// dequant reverses quantize: bucket 0 is the sentinel (-1); buckets [1,10001]
// map back to the original [0,1] grid value k/10000.
func dequant(u uint16) float64 {
	if u == 0 {
		return -1
	}
	return float64(u-1) / 10000
}
```

- [ ] **Step 4: Rodar — deve PASSAR**

Run: `go test ./internal/knn/ -run 'TestQuantizeMapping|TestDequantRoundTrip' -v`
Expected: PASS. (O resto do pacote ainda compila: `dist2`/`topK` seguem uint16/uint64 — `dequant` está só no teste por enquanto.)

- [ ] **Step 5: Commit**

```powershell
git add internal/knn/knn.go internal/knn/knn_test.go
git commit -m "feat(knn): quantize on native 4-decimal grid (x10000) + dequant helper"
```

---

### Task 2: Migrar a distância para float64 (mudança atômica no pacote `knn`)

Uma migração de tipo: a query passa a fluir como `*[Dims]float64` e `topK.dist`/distâncias viram `float64`. O pacote não compila no meio — então todos os arquivos abaixo mudam juntos, depois compila+testa.

**Files:**
- Modify: `internal/knn/knn.go` (`topK`, `newTopK`, `worst`, `consider`, `dist2`, `rowDist2`, `Score`, `bruteScore`, `bruteTopK`)
- Modify: `internal/knn/ivf.go` (`ivfIndex.centroids`, `buildIVF` seed, `trainKMeans`, `nearestCentroid`, `centroidDist`, `search`, `searchTopK`)
- Modify: `internal/knn/vptree.go` (`vpPair.d2`, `build`, `search`, `searchTopK`, `searchNode`)
- Modify: `internal/knn/serialize.go` (`indexVersion` → 3; centróides via float64; +`math` import; +`writeF64Slice`/`readF64Slice`)
- Modify: `internal/knn/knn_search_test.go` (remover `quantizeQuery`; queries float64)

- [ ] **Step 1: `knn.go` — topK float64**

Substituir `topK`/`newTopK`/`worst`/`consider` (linhas ~178-207) por:

```go
// topK is a tiny sorted buffer (ascending distance) of the K nearest seen so far.
type topK struct {
	dist  [K]float64
	fraud [K]bool
}

func newTopK() topK {
	var t topK
	for i := range t.dist {
		t.dist[i] = math.MaxFloat64
	}
	return t
}

// worst is the distance of the current K-th nearest (the pruning bound).
func (t *topK) worst() float64 { return t.dist[K-1] }

// consider inserts (dist, fraud) into the buffer if it beats the current K-th.
func (t *topK) consider(dist float64, fraud bool) {
	if dist >= t.dist[K-1] {
		return
	}
	pos := K - 1
	for pos > 0 && t.dist[pos-1] > dist {
		t.dist[pos] = t.dist[pos-1]
		t.fraud[pos] = t.fraud[pos-1]
		pos--
	}
	t.dist[pos] = dist
	t.fraud[pos] = fraud
}
```

- [ ] **Step 2: `knn.go` — dist2 / rowDist2 float64 com dequant**

Substituir `dist2` (linhas ~152-161) e `rowDist2` (~164-173) por:

```go
// dist2 returns the squared euclidean distance between the un-quantized query q
// and stored row `row`. The stored uint16 ref is dequantized to its exact
// 4-decimal float64, so the result matches the float64 ground-truth distance.
func (ix *Index) dist2(q *[vectorize.Dims]float64, row int) float64 {
	off := row * vectorize.Dims
	data := ix.data
	var dist float64
	for d := 0; d < vectorize.Dims; d++ {
		diff := q[d] - dequant(data[off+d])
		dist += diff * diff
	}
	return dist
}

// rowDist2 is the squared distance between two stored rows (used by the VP-tree
// build). Both rows are dequantized to float64.
func (ix *Index) rowDist2(a, b int) float64 {
	oa, ob := a*vectorize.Dims, b*vectorize.Dims
	data := ix.data
	var dist float64
	for d := 0; d < vectorize.Dims; d++ {
		diff := dequant(data[oa+d]) - dequant(data[ob+d])
		dist += diff * diff
	}
	return dist
}
```

- [ ] **Step 3: `knn.go` — Score sem quantizar a query; brute float64**

Substituir `Score` (linhas ~221-239) e `bruteScore`/`bruteTopK` (~242-255) por:

```go
// Score returns the fraud fraction among the K nearest reference vectors. The
// query is NOT quantized — it stays full-precision float64 and is compared
// against the dequantized refs (exact reproduction of the ground-truth 5-NN).
func (ix *Index) Score(query [vectorize.Dims]float64) float64 {
	if ix.n == 0 {
		return 0
	}
	switch ix.mode {
	case modeVPTree:
		return ix.vp.search(ix, &query)
	case modeIVF:
		return ix.ivf.search(ix, &query)
	default:
		return ix.bruteScore(&query)
	}
}

// bruteScore scans every row — exact, O(n). The fallback and the small-N path.
func (ix *Index) bruteScore(q *[vectorize.Dims]float64) float64 {
	tk := ix.bruteTopK(q)
	return tk.fraudScore()
}

// bruteTopK is the exact K nearest by full scan (source of truth for the
// VP-tree/IVF correctness tests).
func (ix *Index) bruteTopK(q *[vectorize.Dims]float64) topK {
	tk := newTopK()
	for idx := 0; idx < ix.n; idx++ {
		tk.consider(ix.dist2(q, idx), ix.isFraud(idx))
	}
	return tk
}
```

(O bloco em `Score` que criava `var q [Dims]uint16` e chamava `quantize` some — a query crua é passada direto. `Add` continua usando `quantize` para ARMAZENAR. `math` segue importado por `newTopK`.)

- [ ] **Step 4: `ivf.go` — centróides float64**

Substituir o tipo `ivfIndex` (linhas ~19-24) por:

```go
type ivfIndex struct {
	centroids []float64 // nlist * Dims, in the [0,1]/sentinel value space
	lists     [][]int32
	nlist     int
	nprobe    int
}
```

Em `buildIVF`, trocar a alocação dos centróides e o seed (linhas ~51-66) por:

```go
	f := &ivfIndex{
		centroids: make([]float64, nlist*D),
		lists:     make([][]int32, nlist),
		nlist:     nlist,
		nprobe:    nprobe,
	}

	// Seed centroids from evenly-spaced rows (deterministic, spreads them out).
	stride := ix.n / nlist
	if stride < 1 {
		stride = 1
	}
	for c := 0; c < nlist; c++ {
		row := c * stride
		for d := 0; d < D; d++ {
			f.centroids[c*D+d] = dequant(ix.data[row*D+d])
		}
	}
```

- [ ] **Step 5: `ivf.go` — trainKMeans / nearestCentroid float64**

Em `trainKMeans`, trocar a acumulação e a atualização de centróide (linhas ~112-129) por:

```go
		for i, row := range sample {
			c := int(assign[i])
			off := int(row) * D
			base := c * D
			for d := 0; d < D; d++ {
				sum[base+d] += dequant(ix.data[off+d])
			}
			cnt[c]++
		}
		for c := 0; c < f.nlist; c++ {
			if cnt[c] == 0 {
				continue // keep the old centroid for an empty cell
			}
			base := c * D
			for d := 0; d < D; d++ {
				f.centroids[base+d] = sum[base+d] / float64(cnt[c])
			}
		}
```

Substituir `nearestCentroid` (linhas ~195-214) por:

```go
// nearestCentroid returns the index of the centroid closest to row (float64).
func (f *ivfIndex) nearestCentroid(ix *Index, row int) int {
	D := vectorize.Dims
	off := row * D
	data := ix.data
	best := 0
	bestD := math.MaxFloat64
	for c := 0; c < f.nlist; c++ {
		base := c * D
		var dist float64
		for d := 0; d < D; d++ {
			diff := dequant(data[off+d]) - f.centroids[base+d]
			dist += diff * diff
		}
		if dist < bestD {
			bestD = dist
			best = c
		}
	}
	return best
}
```

- [ ] **Step 6: `ivf.go` — centroidDist / search float64**

Substituir `centroidDist` (linhas ~217-226), `search` (~228-231) e `searchTopK` (~233-265) por:

```go
// centroidDist returns the squared distance from query q to centroid c.
func (f *ivfIndex) centroidDist(q *[vectorize.Dims]float64, c int) float64 {
	D := vectorize.Dims
	base := c * D
	var dist float64
	for d := 0; d < D; d++ {
		diff := q[d] - f.centroids[base+d]
		dist += diff * diff
	}
	return dist
}

func (f *ivfIndex) search(ix *Index, q *[vectorize.Dims]float64) float64 {
	tk := f.searchTopK(ix, q)
	return tk.fraudScore()
}

func (f *ivfIndex) searchTopK(ix *Index, q *[vectorize.Dims]float64) topK {
	np := f.nprobe

	// Select the nprobe nearest centroids into stack arrays (no allocation).
	var pd [maxProbe]float64
	var pc [maxProbe]int32
	for i := 0; i < np; i++ {
		pd[i] = math.MaxFloat64
	}
	for c := 0; c < f.nlist; c++ {
		dist := f.centroidDist(q, c)
		if dist >= pd[np-1] {
			continue
		}
		pos := np - 1
		for pos > 0 && pd[pos-1] > dist {
			pd[pos] = pd[pos-1]
			pc[pos] = pc[pos-1]
			pos--
		}
		pd[pos] = dist
		pc[pos] = int32(c)
	}

	// Scan only the chosen cells.
	tk := newTopK()
	for i := 0; i < np; i++ {
		for _, row := range f.lists[pc[i]] {
			tk.consider(ix.dist2(q, int(row)), ix.isFraud(int(row)))
		}
	}
	return tk
}
```

- [ ] **Step 7: `vptree.go` — distâncias float64**

Trocar `vpPair` (linhas ~29-32) por:

```go
type vpPair struct {
	id int32
	d2 float64
}
```

Em `build`, a linha que calcula `medianD2` (linha ~79) e o `radius` (~86) continuam válidas com `d2 float64` (`sort.Slice` já compara `sc[i].d2 < sc[j].d2`; `radius := float32(math.Sqrt(float64(medianD2)))` vira `radius := float32(math.Sqrt(medianD2))`). Substituir a linha 86:

```go
	radius := float32(math.Sqrt(medianD2))
```

Substituir `search`/`searchTopK`/`searchNode` (linhas ~95-138) por:

```go
func (t *vpTree) search(ix *Index, q *[vectorize.Dims]float64) float64 {
	tk := t.searchTopK(ix, q)
	return tk.fraudScore()
}

func (t *vpTree) searchTopK(ix *Index, q *[vectorize.Dims]float64) topK {
	tk := newTopK()
	t.searchNode(ix, q, t.root, &tk)
	return tk
}

func (t *vpTree) searchNode(ix *Index, q *[vectorize.Dims]float64, ni int32, tk *topK) {
	if ni < 0 {
		return
	}
	point := t.nodes[ni].point
	d2 := ix.dist2(q, int(point))
	tk.consider(d2, ix.isFraud(int(point)))

	inner, outer := t.nodes[ni].inner, t.nodes[ni].outer
	if inner < 0 && outer < 0 {
		return
	}

	// Pruning uses TRUE distances (triangle inequality fails for squared dists).
	d := math.Sqrt(d2)
	r := float64(t.nodes[ni].radius)

	if d < r {
		t.searchNode(ix, q, inner, tk)
		tau := math.Sqrt(tk.worst())
		if d+tau >= r {
			t.searchNode(ix, q, outer, tk)
		}
	} else {
		t.searchNode(ix, q, outer, tk)
		tau := math.Sqrt(tk.worst())
		if d-tau <= r {
			t.searchNode(ix, q, inner, tk)
		}
	}
}
```

- [ ] **Step 8: `serialize.go` — version 3 + centróides float64**

Trocar a constante (linhas ~26-29):

```go
const (
	indexMagic   uint32 = 0x52464958 // "RFIX"
	indexVersion uint32 = 3          // v3: refs uint16 on x10000 grid, float64 distance, float64 centroids (v2 was x65534 uint16/uint64)
)
```

Adicionar `"math"` ao import block (linhas ~16-24): incluir a linha `"math"`.

Em `(*ivfIndex).writeTo` trocar a escrita de centróides (linha ~131):

```go
	if err := writeF64Slice(w, f.centroids); err != nil {
		return err
	}
```

Em `readIVF` trocar a leitura (linha ~162):

```go
	centroids, err := readF64Slice(r)
	if err != nil {
		return nil, err
	}
```

Adicionar dois encoders/decoders (junto dos outros, ex. após `writeU64Slice` e `readU64Slice`):

```go
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
```

- [ ] **Step 9: `knn_search_test.go` — queries float64 (remover `quantizeQuery`)**

Remover a função `quantizeQuery` (linhas ~47-53). Em `TestVPTreeExactMatchesBrute`, substituir o loop (linhas ~70-77) por:

```go
	for i, q := range randomQueries(300, 99) {
		q := q // addressable per-iteration copy
		want := ref.bruteTopK(&q)
		got := vp.vp.searchTopK(vp, &q)
		if got.dist != want.dist {
			t.Fatalf("query %d: VP distances %v != brute %v", i, got.dist, want.dist)
		}
	}
```

Em `TestIVFRecallHigh`, substituir o loop (linhas ~95-101) por:

```go
	queries := randomQueries(300, 123)
	var totalRecall float64
	for _, q := range queries {
		q := q
		want := ref.bruteTopK(&q)
		got := ivf.ivf.searchTopK(ivf, &q)
		totalRecall += recallAtK(want, got)
	}
```

(`recallAtK` compara `got.dist[j] == want.dist[i]` — float64, comparação exata válida pois é a mesma computação determinística.)

- [ ] **Step 10: Compilar o pacote inteiro**

Run: `go build ./... ; go vet ./internal/knn/`
Expected: sem erros. Se houver “cannot use … (uint16) as float64” sobrou um ponto não migrado — corrigir até build limpo.

- [ ] **Step 11: Rodar os testes do pacote**

Run: `go test ./internal/knn/ -v`
Expected: PASS — incluindo `TestVPTreeExactMatchesBrute` (VP == brute), `TestIVFRecallHigh` (recall ≥ 0.90), `TestIVFRoundTrip` (centróides float64 round-trip + score igual), `TestScore*`, `TestSentinelMatchesSentinel`.

- [ ] **Step 12: Rodar TODOS os testes**

Run: `go test ./...`
Expected: tudo `ok` (cmd/diag, cmd/buildindex, handler_test, vectorize, dataset). `cmd/diag` usa `Index.Score(float64)` (assinatura inalterada) — compila sem mudança.

- [ ] **Step 13: Commit**

```powershell
git add internal/knn/
git commit -m "feat(knn): float64 distance vs un-quantized query; centroids float64; index v3"
```

---

### Task 3: Rebuildar o `index.bin` na nova escala/formato (v3)

**Files:** nenhum de código; gera `index.bin` (gitignored).

**Pré:** `resources/references.json.gz` presente (gitignored, ~48MB).

- [ ] **Step 1: Buildar o índice v3 offline**

Run:
```powershell
go run ./cmd/buildindex
```
Expected: log de load + kmeans, grava `index.bin`. (Se `cmd/buildindex` aceita flags/env de saída diferentes, conferir seu `main.go`; rodar sem args usa os defaults.) Confirmar que o arquivo foi regenerado:
```powershell
Get-Item .\index.bin | Select-Object Length, LastWriteTime
```
Expected: `LastWriteTime` agora; `Length` ~90MB (uint16 data + float64 centroids).

- [ ] **Step 2: Sanidade — carregar e pontuar (sem rebuild)**

Run: `go test ./internal/knn/ -run TestIVFRoundTrip -v`
Expected: PASS (garante que Save/LoadIndex v3 é consistente). Commit não necessário (index.bin é gitignored).

---

### Task 4: Medir offline com `cmd/diag` (portão de detecção)

**Files:** opcional — atualizar rótulo cosmético em `cmd/diag/main.go` (linha ~127 `{"u16-brute    ", u8Cnt}`).

**Pré:** `resources/references.json.gz` + `../rinha-de-backend-2026/test/test-data.json`.

- [ ] **Step 1: (Opcional) renomear o rótulo do método brute pra não enganar**

Em `cmd/diag/main.go` linha ~127, trocar `{"u16-brute    ", u8Cnt}` por `{"u16/f64-brute", u8Cnt}` (o brute agora é float64-dist sobre refs ×10000). Cosmético.

- [ ] **Step 2: Rodar o diagnóstico (~vários min, full CPU)**

Run:
```powershell
go run ./cmd/diag
```
Expected: tabela `DIAGNÓSTICO DE FAILURES`. **Critério de sucesso do Caminho A:**
- linha `float64-exact` → failures **0** (oráculo, sempre 0).
- linha brute (`u16/f64-brute`) → failures **~0** (a tese: dequant exato reproduz o float64). Antes (×65534) era 144.
- linha `IVF (4096/12)` → failures = só **recall** (dezenas, ex. ~30-40); antes 180.

- [ ] **Step 3: Registrar no ledger — criar `docs/performance/08-quantizacao-grade-nativa.md`**

Criar o doc seguindo o padrão da trilha (ver `docs/performance/_template.md` e `07-diagnostico-failures.md`): pergunta, hipótese (query-quantization dominava os 144; refs 4-casas cabem lossless ×10000; distância float64 = ground truth), **tabela de resultados do `cmd/diag`** (failures por método antes/depois), e veredito. Colar os números reais medidos no Step 2.

- [ ] **Step 4: Commit**

```powershell
git add docs/performance/08-quantizacao-grade-nativa.md cmd/diag/main.go
git commit -m "docs(perf): 08 quantizacao grade nativa x10000 (diag: brute ~0, IVF recall-only)"
```

---

### Task 5: Medir sob carga no docker (`run-test.ps1`) — portão de p99/final

**Files:** nenhum.

- [ ] **Step 1: Subir stack local + k6 + nota**

Run:
```powershell
.\run-test.ps1 -SkipSmoke
```
Expected: builda imagem local (com o `index.bin` v3), sobe api1/api2/nginx sob os caps, roda k6, imprime `p99 / failure_rate / final_score`. **Esperado:** `Err=0` mantido; `failure_rate` cai (detecção melhor, IVF recall-only); `p99` ~igual (~387ms local; distância float64 não muda volume de varredura); `final_score` sobe de **1843** para ~2700-3000 (det melhor; p99 ~igual).

- [ ] **Step 2: Registrar a linha medida no ledger**

Adicionar uma linha na tabela de progresso de `docs/performance/README.md` (etapa 08, ambiente `docker-9999`) e fechar o veredito do doc 08 com p99/failure_rate/p99_score/detection_score/final_score reais.

- [ ] **Step 3: Commit**

```powershell
git add docs/performance/README.md docs/performance/08-quantizacao-grade-nativa.md
git commit -m "docs(perf): registra final_score do Caminho A sob carga (docker-9999)"
```

---

## Decisão pós-Caminho-A (gate pro Caminho C)

Com os números do Task 4/5 na mão:
- Se `final_score` do Caminho A ≥ 1843 → **candidato a publicar** (não publicar ainda; aguardar comparar com o modelo).
- Independente do resultado, seguir pro **Caminho C (GBDT)** pra atacar o p99 (gap dominante). O plano do Caminho C é escrito DEPOIS, com os números do A medidos, inspecionando um dump real do xgboost (pra zero placeholder no parsing em Go).
- Promover na branch `submission` só o `SCORER` (knn ×10000 vs model) com maior `final_score`.

## Notas de execução

- **Commits:** o usuário pediu "implementa". Confirmar com ele antes do primeiro commit se quiser revisar; os passos de commit acima são a disciplina TDD do plano.
- **Não tocar** `docker-compose.yml` da branch `submission` nem republicar imagem nesta fase — só medição local.
- `Err` deve seguir **0** em qualquer caminho (fallback rápido preservado em `main.go`).
