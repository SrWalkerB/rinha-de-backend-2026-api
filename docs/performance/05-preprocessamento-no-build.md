# 05 - Pré-processamento no build

> **Pré-requisito:** [04 - Tuning de runtime](04-tuning-runtime.md) · **Estimativa:** ~2h · **Dificuldade:** ★★★
> Status: **✅ implementado.** É o que derruba o startup (de ~85s pra ~25ms) e destrava `nlist` grande.

## Conceito

Hoje o índice IVF é construído **no startup**, dentro do container (0.45 CPU). Isso tem dois
custos:
1. **Startup lento** — o k-means + atribuição dos 3M leva ~85s (a `/ready` só responde 200
   depois).
2. **Limita o `nlist`** — construir um `nlist` grande (que daria query mais rápida → p99
   menor) é lento demais sob o cap de CPU.

A Rinha **permite** (e recomenda) pré-processar os arquivos de referência no **build** da
imagem. Lá não tem cap de CPU — dá pra usar a máquina inteira, com `nlist` grande, e **gravar
o índice pronto** dentro da imagem. No startup, só carrega (rápido).

> "Quanto mais processamento sair do runtime, melhor tende a ficar o p99." — AVALIACAO.md

## Por que importa aqui

É o que destrava o `p99`. Com o índice pré-construído:
- **Startup quase instantâneo** (carrega bytes em vez de rodar k-means).
- **`nlist` grande livre** (ex.: 8192) → células minúsculas → query sub-0.1ms → capacidade
  muito acima de 900 rps → a fila some → **p99 cai de ~1200ms pra single/double-digit ms** →
  `p99_score` salta de ~-80 pra +1000~+2000.

## Como ficou (implementado)

1. **Serializar o índice** — `internal/knn/serialize.go`: `Save(path)` / `LoadIndex(path)`.
   Formato binário próprio (sem `encoding/gob`, que é lento/reflexivo): cabeçalho com
   magic + versão + `dims` + `mode`, depois `n`, `data` (uint8), `bits` (uint64), e — pro
   IVF — `nlist`, `nprobe`, `centroids`, os tamanhos das `lists` e todos os ids `int32`
   achatados. O leitor remonta `lists [][]int32` como fatias de **um único array** contíguo
   (menos alocações). Slices gravadas como blocos de bytes little-endian (uma `Write` por
   slice), não elemento a elemento.
2. **Ferramenta de build** — `cmd/buildindex/main.go`: lê `references.json.gz`, roda
   `dataset.Load` + `ix.Build({ivf, KNN_NLIST})` e grava `index.bin`. Tudo por env
   (`REFERENCES_PATH`, `INDEX_OUT`, `KNN_NLIST`, `KNN_KMEANS_ITERS`).
3. **Build paralelo** — as duas etapas caras do k-means (`assignAll` e a atribuição por
   iteração) agora rodam em todos os cores via `parallelFor` (cada índice escrito uma vez →
   sem lock; divisão determinística → índice reproduzível). No build offline isso usa a
   máquina inteira; no fallback de runtime (`GOMAXPROCS=1`) cai pra execução direta.
4. **Dockerfile (multi-stage)** — o estágio de build compila `buildindex`, roda ele com CPU
   cheia (sem cap) e produz `/out/index.bin`; o estágio final copia **só** `index.bin` (o
   `.gz` não entra na imagem de runtime). `ENV INDEX_PATH=/app/index.bin`. `nlist` é um
   `ARG KNN_NLIST` (default 4096), exposto no `docker-compose.yml` em `build.args`.
5. **Runtime** — `main.go`: se `INDEX_PATH` está setado, `knn.LoadIndex` e pula o `Build`;
   `KNN_NPROBE` retuna o índice no startup (via `SetNProbe`) **sem rebuild**. Se o arquivo
   faltar/corromper, cai no caminho antigo (carrega `.gz` + constrói) — sem crash.
6. **Memória** — `index.bin` ≈ 52MB (data ~42MB + ids ~12MB + centroids/bits). Lido por
   `bufio` simples, cabe folgado no `GOMEMLIMIT`.

## Construir o índice localmente

```powershell
$env:Path = "C:\Program Files\Go\bin;" + $env:Path
$env:REFERENCES_PATH = ".\resources\references.json.gz"
$env:INDEX_OUT = ".\resources\index.bin"
$env:KNN_NLIST = "4096"
go run ./cmd/buildindex
```

No Docker isso roda dentro do `docker build` (estágio de build, CPU cheia) — você não precisa
rodar à mão; o `index.bin` é embutido na imagem automaticamente.

## Como medir

```powershell
.\run-test.ps1 -SkipSmoke
```
Esperado: `/ready` em **milissegundos** (não ~85s) **e** `p99` despencando (nlist grande).
Registrar a linha na tabela — provavelmente o maior salto de `p99_score` da trilha.

## Armadilhas

- **Versão do índice** — se mudar o formato, invalide `index.bin` (um magic number/versão no
  cabeçalho).
- **Determinismo** — o k-means usa amostra determinística; bom, o índice é reproduzível.
- **Tamanho da imagem** — `index.bin` (~54MB) entra na imagem; ok, é compatível com o limite.
- **Não esquecer o fallback** — sem `index.bin`, ainda tem que subir (constrói no startup).

## Registro

**Build offline medido (local, 32 cores, 2026-05-30):**
- carregar `references.json.gz` (3M vetores): **6.7s**
- k-means `nlist=4096`, 8 iterações (paralelo): **10.4s**
- `index.bin` gravado: **51.9 MB**
- **startup carregando o índice pronto: 25ms** (vs ~85s construindo no runtime sob cap)
- sanidade na API real: exemplo legítimo → `{"approved":true,"fraud_score":0}`; exemplo
  fraude → `{"approved":false,"fraud_score":1}` ✅

**Sob carga real medido (`docker-9999`, 2026-05-30):**
- `p99`: **1202ms → 484ms** (`p99_score` -80 → **+315**)
- `final_score`: +1136 → **+1531** (Δ **+395**)
- `Err: 0` (100% servido), `detection_score` ~igual (+1216)
- o `failure_rate: 0.5%` é **erro de classificação** (FP=147 + FN=119 de 53.595), não timeout

Ou seja: o ganho do passo 05 veio **todo do p99** (células ~4× menores que o nlist=1024 anterior).
Pra ir além: `nlist=8192` (build args) ou SIMD (passo 06).

> **Nota de submissão:** `GOAMD64=v3` (passo 06) NÃO foi ligado no `Dockerfile` de propósito —
> exige AVX2 na CPU de avaliação (risco de `SIGILL` se não tiver). Fica como experimento
> opcional no passo 06.

## Próximo

[06 - SIMD (avançado)](06-simd-avancado.md): o último quilômetro do cálculo de distância.
