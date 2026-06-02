# CLAUDE.md — rinha-fraud (Rinha de Backend 2026)

API de **detecção de fraude por busca vetorial**, em Go puro (stdlib, sem deps). Submissão para
a competição Rinha de Backend 2026.

## Objetivo

Maximizar o `final_score` da Rinha (`p99_score + detection_score`, faixa −6000..+6000):
1. **Atender 100% das requisições** sob carga (900 req/s) — ✅ `Err=0`, sustenta a carga.
2. **p99 baixo** — busca ~219µs/query de CPU (nprobe=16); ver caveat do p99 local abaixo.
3. **Failures de classificação mínimas** — E=30 (0.03%) @ nprobe=16; tunável via nprobe.

Meta de aprendizado do dono: usar o projeto pra **aprender performance/otimização** medindo
cada técnica. Princípio: **meça antes de complicar.**

## O desafio (regras essenciais)

- **Endpoints (porta 9999, atrás do nginx):** `GET /ready` (200 quando carregado), `POST
  /fraud-score` (transação → `{approved, fraud_score}`).
- **Detecção (gabarito):** vetoriza payload em **14 dims** normalizadas → **5-NN exato float64
  brute force** sobre 3M vetores rotulados → `fraud_score = nº_fraudes/5`, `approved = score < 0.6`.
- **failures = FP + FN + Err.** Mantemos `Err=0` → toda failure é classificação.
- **Scoring (AVALIACAO.md):**
  - `p99_score = 1000·log10(1000/max(p99,1))` — teto **+3000** em ≤1ms, corte **−3000** acima de 2000ms.
  - `detection_score = 1000·log10(1/max(ε,0.001)) − 300·log10(1+E)`, `E = 1·FP+3·FN+5·Err`, `ε=E/N`.
    **O componente de taxa satura em +3000 quando ε≤0.001 (E≤~54)** → daí `det = 3000 − 300·log10(1+E)`,
    então **minimizar E é tudo.** Teto +3000 só com E=0.
  - `final = p99_score + detection_score`.

## Arquitetura (busca: IVF de células contíguas)

A detecção (5-NN exato) é PORTÁVEL e quase resolvida; **o gargalo era latência**. 5-NN exato é
inviável sub-ms aqui (dados "espalhados": 5ª-NN a ~0.5 euclidiano em ~10 dims → varre raio grande).
A solução: **IVF aproximado com memória CONTÍGUA** (cache-friendly), que o voto-maioria no limiar
0.6 tolera bem.

1. `internal/vectorize` — payload → `[14]float64` (normalizado [0,1]; sentinela −1 nas dims 5/6
   quando `last_transaction` é null). **Fiel ao gabarito** (float64-exato = 0 failures no diag).
   `Parse(normJSON, mccJSON)` constrói o Vectorizer (usado por main + diag).
2. `internal/index` — **o coração.** quantiza cada dim em **uint16** na grade nativa de 4 casas
   (`round(v*10000)+1`; −1→0); distância **float64** com a query NÃO-quantizada vs refs dequantizadas
   (table lookup `dequantTab`, sem divisão no hot loop). Estrutura em 2 níveis:
   - **16 buckets duros** (`bucketOf`): bits de `is_online`(d9), `card_present`(d10),
     `unknown_merchant`(d11) + flag-de-nulo(d5/d6). São as únicas dims com gap ≥1.0 → partição exata.
   - **k-means por bucket** em `nlist` células, linhas **armazenadas contíguas por célula** (CSR
     `cellStart`). Query: `bucketOf` → varre as `nprobe` células de centróide mais próximo
     (contíguo, prefetch) + buckets vizinhos só se o `bucketPenalty` (≥1.0) admitir. Hot path
     zero-alloc, buffers na stack (`maxProbe=64`).
3. `internal/dataset` — `Load(path, cap, nlist, iters)` faz stream do `references.json.gz` (3M) pro
   `index.Builder`.
4. Índice **pré-construído no Docker build** (`cmd/buildindex` → `index.bin` embutido, ~75s k-means);
   startup só carrega (~ms). `main.go` carrega de `INDEX_PATH`, fallback build no startup.

## Estado / curva de sintonia (medido no dado real, 54.100 entradas)

`detection_score` é **portável**; latência varia por máquina. Sweep de `nprobe` (diag offline):

| nprobe | E | failures | det_score | compute/query (dev) |
|---|---|---|---|---|
| 8  | 91 | 0.087% | +2185 | 122µs |
| **16** | **31** | **0.031%** | **+2552** | **219µs** |
| 32 | 14 | 0.011% | +2647 | 379µs |
| 64 |  4 | 0.004% | +2790 | 840µs |

`go test ./...` verde; exatidão provada (com `nlist=1`/`nprobe=nlist` o index == brute float64).

## Aprendizados-chave (NÃO re-descobrir)

- **Não é o Go.** Submissão Go no rank faz 0.445ms/0%/6000. Gargalo = padrão de memória + algoritmo.
- **5-NN EXATO é inviável sub-ms aqui.** Dados espalhados (5ª-NN ~0.5) → kd-tree/brute varrem
  250k–1.17M linhas/query (medido). Detecção exata = E=0 mas p99 catastrófico.
- **kd-tree COLAPSA sob carga: acesso ESPALHADO** (reorder espalha linhas) → cache miss/linha
  (~100ns) → sob concorrência thrasha o cache → k6 deu −6000 (84% timeout). Localidade de cache
  domina o nº de linhas.
- **IVF de células CONTÍGUAS resolve:** ~219µs/query (nprobe=16), sustenta 900 req/s (Err=0),
  E=30. Contíguo = prefetch, fica em L2.
- **⚠️ O p99 do k6 LOCAL é artefato da rede do Docker Desktop Windows.** Provado: nprobe=1 (29µs,
  2% CPU) e nprobe=16 (219µs) dão p99 IDÊNTICO (~725ms). Compute a 2% de util não satura — é a VM.
  **Pra medir p99 real → teste de prévia da Rinha (roda no Mac Mini Linux nativo).**
- **Caminho pro 6000:** com IVF o det teto é ~+2790 (E≈4). Pra E→0 (det +3000), considerar
  `nprobe` adaptativo: re-buscar com nprobe alto só as queries com `fraud_score` perto de 0.6
  (fronteira), mantendo o custo médio baixo. (Não implementado.)

## Restrições (container apertado)

- **Mac Mini Late 2014:** 2 cores, 2.6 GHz (Haswell, AVX2). Soma de TODOS os serviços ≤ 1 CPU e 350 MB.
- api1/api2 = 0.45 CPU + 155 MB cada; nginx = 0.10 + 40 MB. `GOMAXPROCS=1`, `GOMEMLIMIT=150MiB`, `GOGC=off`.
- **Memória:** live set ~88 MB (uint16 data 84MB + centróides 1.8MB + bitset + CSR). Cabe folgado.
- Binário estático (`CGO_ENABLED=0`, distroless). Sem deps.

## Arquivos-chave

- `main.go` — servidor; `loadVectorizer` (embed) + carrega `INDEX_PATH` (fallback build); aplica
  `INDEX_NPROBE`/`INDEX_MAX_SCAN`.
- `internal/index/{index,build,serialize}.go` — IVF: busca (index.go), k-means + counting-sort de
  células (build.go), formato binário v3 (serialize.go). `BruteScore`/`ScoreScan` = só diagnóstico.
- `internal/vectorize/{vectorize,load}.go` — vetorização fiel + `Parse`.
- `cmd/buildindex/main.go` — builda `index.bin` offline (Docker build).
- `cmd/diag/main.go` — **harness offline:** replay das 54.100 entradas; compara IVF vs brute float64
  (oráculo), reporta E/det + estatísticas de scan/latência. Knobs via env (`INDEX_NPROBE` etc).
- `Dockerfile` — multi-stage; builda índice no build, embute `index.bin` (sem `.gz` no runtime).
- `docker-compose.yml` — stack de DEV (build local, usado pelo run-test.ps1). `submission/` — branch submission.

## Comandos (Go em `C:\Program Files\Go\bin`, NÃO está no PATH)

```powershell
$env:Path = "C:\Program Files\Go\bin;" + $env:Path
go test ./...                                   # testes (todos verdes)
go run ./cmd/buildindex                          # gera resources/index.bin (~75s)
$env:INDEX_PATH=".\resources\index.bin"; $env:INDEX_NPROBE="16"; go run ./cmd/diag   # diag offline (carrega index, rápido)
.\run-test.ps1 -SkipSmoke                        # builda imagem local + sobe stack + k6 (p99 local = artefato Windows!)
.\publish.ps1 -User srwalkerb                    # push pra Docker Hub
```

## Variáveis de ambiente

| Var | Default | Onde |
|---|---|---|
| `INDEX_PATH` | (vazio) | runtime — carrega índice pronto; senão fallback build |
| `INDEX_NPROBE` | 8 (compose: 16) | runtime — células IVF varridas/bucket/query (retuna SEM rebuild) |
| `INDEX_NLIST` | 1024 | build — células k-means por bucket |
| `INDEX_KMEANS_ITERS` | 10 | build — iterações do k-means |
| `INDEX_MAX_SCAN` | 0 (ilimitado) | runtime — teto de linhas/query (guarda de cauda) |
| `GOMAXPROCS/GOMEMLIMIT/GOGC` | 1 / 150MiB / off | compose |
| `ADDR` | :8080 | porta da API (nginx expõe 9999) |
| `PPROF_ADDR` | (vazio) | debug — NUNCA setar na submissão |

## Submissão (fluxo)

- **2 branches:** `main`/feature (fonte completa) + `submission` (só `docker-compose.yml` por IMAGEM
  pública + `nginx.conf` + `info.json`, **sem fonte**). Repo deve ser **público**.
- A engine puxa do GitHub (branch submission) + Docker Hub (imagem). Use **TAG nova** a cada push +
  `pull_policy: always` (já no submission compose) pra evitar cache.
- Teste de prévia: issue em `zanfranceschi/rinha-de-backend-2026` com `rinha/test`. Final: automático,
  **deadline 2026-06-05**.

## Próximo passo

1. **Push da imagem IVF** + atualizar `submission/` (tag nova) → **teste de prévia** pra medir o p99
   REAL no Mac Mini (o local não serve).
2. Sintonizar `nprobe` (16→32→64) via prévias: subir até o p99 do Mac chegar perto de 1ms (det maior).
3. (Rumo a 6000) `nprobe` adaptativo nas queries de fronteira (score ~0.6) pra cravar E→0.
