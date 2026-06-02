# CLAUDE.md — rinha-fraud (Rinha de Backend 2026)

API de **detecção de fraude por busca vetorial**, em Go puro (stdlib, sem deps). Submissão para
a competição Rinha de Backend 2026.

## Objetivo

Maximizar o `final_score` da Rinha (`p99_score + detection_score`, faixa −6000..+6000):
1. **Atender 100% das requisições** sob carga (900 req/s) — ✅ `Err=0`, sustenta a carga.
2. **p99 baixo** — ⚠️ **GARGALO ATUAL.** Prévia real no Mac Mini (2026-06-02, img 2.3, nprobe=16):
   **p99=51ms → p99_score=1290** (teto +3000 só ≤1ms). NÃO era artefato; Go sem SIMD + GC + CFS = 50x devagar.
3. **Failures de classificação mínimas** — E=31 (det_score=2548) @ nprobe=16; tunável via nprobe.

**Score de prévia atual: 3839** (p99 1290 + det 2548). Topo do rank (C/C++/Rust) crava **6000**: ver Aprendizados.

Meta de aprendizado do dono: usar o projeto pra **aprender performance/otimização** medindo
cada técnica. Princípio: **meça antes de complicar.**

## O desafio (regras essenciais)

- **Endpoints (porta 9999, atrás do nginx):** `GET /ready` (200 quando carregado), `POST
  /fraud-score` (transação → `{approved, fraud_score}`).
- **Detecção (gabarito):** vetoriza payload em **14 dims** normalizadas → **5-NN exato float64
  brute force** sobre 3M vetores rotulados → `fraud_score = nº_fraudes/5`, `approved = score < 0.6`.
- **failures = FP + FN + Err.** Mantemos `Err=0` → toda failure é classificação.
- **⛔ REGRA DO DONO: NÃO olhar código de outros participantes.** Solução 100% própria. Pode comparar
  **scores/p99/linguagem** das prévias (números públicos) pra calibrar meta, mas **nunca ler/clonar
  repos alheios** nem copiar estratégia de implementação. Aprendizado vem de medir o nosso.
- **Scoring (AVALIACAO.md):**
  - `p99_score = 1000·log10(1000/max(p99,1))` — teto **+3000** em ≤1ms, corte **−3000** acima de 2000ms.
  - `detection_score = 1000·log10(1/max(ε,0.001)) − 300·log10(1+E)`, `E = 1·FP+3·FN+5·Err`, `ε=E/N`.
    **O componente de taxa satura em +3000 quando ε≤0.001 (E≤~54)** → daí `det = 3000 − 300·log10(1+E)`,
    então **minimizar E é tudo.** Teto +3000 só com E=0.
  - `final = p99_score + detection_score`.

## Arquitetura (ATUAL — SIMD SoA int16 + refine exato + LB próprio L4)

**Sugestão de arquitetura em uso (revisado 2026-06-02):** o gabarito é 5-NN exato float64; a detecção
é PORTÁVEL. O gargalo é **latência sob carga = SATURAÇÃO (ρ→1)**, não compute cru (single-query ~0.5ms
no Mac, mas system p99 51ms = amplificação de fila/throttle). Dois eixos atacam ρ e COMPÕEM:
**(A) menos CPU/query** (SIMD) e **(B) mais CPU pra API** (LB leve). Go não tem SIMD nem autovec, então
o kernel de distância vai em **assembly Plan9 AVX2 à mão** (`simd_amd64.s`, CGO off, sem deps, fallback
escalar guardado por CPUID — nunca SIGILL).

1. `internal/vectorize` — payload → `[14]float64` (normalizado [0,1]; sentinela −1 nas dims 5/6 quando
   `last_transaction` é null). Fiel ao gabarito (float64-exato = 0 failures no diag).
2. `internal/index` — **o coração.** uint16 na grade nativa de 4 casas (`round(v*10000)+1`; −1→0).
   - **16 buckets duros** (`bucketOf`, dims 9/10/11 + flag-nulo 5/6, gap ≥1.0 → partição exata) ×
     **k-means em `nlist=1024` células/bucket**.
   - **Layout SoA (dim-major por bucket)** [v4]: dentro do range de linhas do bucket, guarda todas as
     dim 0, depois dim 1, ... (+`simdTailPad`=8). Permite o kernel carregar 8 linhas de uma dim por
     `VMOVZXWD`. `cellStart` segue em linhas globais (células = sub-ranges locais contíguos por dim).
   - **Busca**: quantiza a query → `qcode[14]int32` (1×). `bucketOf` → varre os 1024 centróides
     (escalar float64, escolhe `nprobe` células). Por célula: **kernel int16 SoA** `distSoAi16AVX2`
     (8 linhas/YMM: `VPSUBW`→`VPMADDWD`-style `VPSUBD`/`VPMULLD`/`VPADDD`) calcula distâncias INTEIRAS
     (filtro barato; sentinela natural pois `quantize(-1)=0`). Coleta os `candK=64` de menor distância
     inteira (`intTopK`) + poda cross-bucket conservadora (`penalty*1e8 ≥ worst+1e5`). No fim,
     **refine float64 EXATO** só nos ≤64 candidatos → 5-NN exato. → **detecção (E) idêntica ao escalar**
     (o refine é exato); o int16 só filtra. Hot path zero-alloc (buffers na stack).
   - SIMD medido: per-row asm e batch-AoS são DUDS (latency-bound, perdem pro escalar inlined); **SoA +
     8-linhas-paralelas é o que ganha** (kernel 4.4×; integrado mean 155→82µs = 1.9×). p99 ainda é
     **centroid-bound** (varredura escalar O(nlist) por bucket nas queries cross-bucket) → próximo corte.
3. `cmd/lb` — **load balancer próprio L4** (TCP splice, `io.Copy`=splice(2) no Linux, zero-parse HTTP)
   no lugar do nginx. nginx (L7) precisava ~0.30 CPU pra não se auto-throttlar; o L4 sobrevive em 0.10
   → devolve 0.20 pras APIs (CPU-bound). Satisfaz "≥1 LB + ≥2 instâncias" (é um LB, só não-nginx).
   Topologia: **lb 0.10 + api 0.45×2 = 1.00 CPU / 350MB.** Mesma imagem, entrypoint `/app/lb`.
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

**Prévia real (Mac Mini, 2026-06-02, img `srwalkerb/rinha-fraud:2.3`, commit 436fb82, nprobe=16):**
p99=**51.18ms** (p99_score 1290.9), E=31 (FP=10/FN=7/Err=0), det_score 2548.5, **final=3839.4**.
→ p99 domina o gap pro 6000 (precisa 51ms→~1ms = +1710; det adaptativo rende só ~+450).

## Aprendizados-chave (NÃO re-descobrir)

- **É PARCIALMENTE o Go (revisado 06-02).** Go pode chegar a 6000 (ref 0.445ms/0%), MAS exige fight
  contra (a) ausência de autovec — kernel de distância 8–16x mais lento que C/Rust com AVX2 (commit
  `0d08c3b`); (b) tail de GC; (c) `GOMAXPROCS=1` sob CFS. Líderes da prévia = C/C++/Rust justamente
  porque pegam SIMD+sem-GC de graça → cravam p99 sub-ms **e** E=0. Gargalo = memória + algoritmo +
  **vetorização do hot loop**.
- **5-NN EXATO É viável sub-ms — com SIMD (revisado 06-02).** Líder em **Rust** fez exato/E=0 em
  **0.36ms** no mesmo Mac. Antes concluímos "inviável" medindo **em Go-sem-SIMD** (250k–1.17M linhas,
  ~100ns/miss). AVX2/FMA × 2 cores Haswell ≈ 40 GFLOP/s → 3M×14 diffs ~1–2ms brute, sub-ms com índice
  leve. **Não generalizar conclusão de latência tirada sem SIMD.**
- **kd-tree COLAPSA sob carga: acesso ESPALHADO** (reorder espalha linhas) → cache miss/linha
  (~100ns) → sob concorrência thrasha o cache → k6 deu −6000 (84% timeout). Localidade de cache
  domina o nº de linhas.
- **IVF de células CONTÍGUAS resolve:** ~219µs/query (nprobe=16), sustenta 900 req/s (Err=0),
  E=30. Contíguo = prefetch, fica em L2.
- **⚠️ O p99 do k6 LOCAL é artefato da rede do Docker Desktop Windows** (não confiar nele). MAS o
  p99 REAL no Mac Mini também é alto: **51ms** (prévia 06-02). Não confundir: local ~725ms = VM
  Windows; 51ms = compute Go real devagar. **Pra p99 real → só prévia da Rinha (Mac Mini Linux).**
- **Caminho pro 6000 (revisado):** o gap dominante é **p99 (1290→3000 = +1710)**, não det
  (+450). Prioridade: **vetorizar o hot loop** (SIMD via `avo`/Plan9-asm ou `unsafe`) pra recuperar
  o 8–16x perdido — único jeito de Go aproximar do p99 dos líderes. Com SIMD, reavaliar brute exato
  (mata E **e** p99 juntos, igual o líder Rust 0.36ms). `nprobe` adaptativo na fronteira (score ~0.6) = ganho
  secundário pra E→0. (Nada implementado.)

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

## Próximo passo (revisado 2026-06-02 — prévia deu 3839, gargalo = p99 51ms)

1. **Vetorizar o hot loop (SIMD/AVX2).** Maior alavanca: p99 51ms→~1ms = +1710. Distância em
   `internal/index/index.go` via `avo`/Plan9-asm ou `unsafe`+lanes. Recupera o 8–16x que Go perde
   sem autovec. Medir compute/query antes/depois no diag.
2. **Com SIMD, reavaliar brute exato** (sem IVF): pode cravar E=0 **e** p99 sub-ms de uma vez (prévia
   provou que dá: líder fez 0.36ms/E=0). Se viável, mata o trade-off do nprobe.
3. Secundário: `nprobe` adaptativo na fronteira (score ~0.6) pra E→0 (det +450); só depois do p99.
