# CLAUDE.md — rinha-fraud (Rinha de Backend 2026)

API de **detecção de fraude por busca vetorial**, em Go puro (stdlib, sem deps). Submissão para
a competição Rinha de Backend 2026.

## Objetivo

Maximizar o `final_score` da Rinha (`p99_score + detection_score`, faixa −6000..+6000):
1. **Atender 100% das requisições** sob carga (900 req/s) — ✅ feito (`Err=0`, zero timeout).
2. **p99 baixo** — ✅ 387ms local (de 2002ms).
3. **0 failures de classificação** — 🔄 em 0.33% (de 0.5%); rumo a 0.

Meta de aprendizado do dono: usar o projeto pra **aprender performance/otimização** medindo
cada técnica (trilha em `docs/performance/`). Princípio: **meça antes de complicar.**

## O desafio (regras essenciais)

- **Endpoints (porta 9999, atrás do nginx):** `GET /ready` (200 quando carregado), `POST
  /fraud-score` (recebe transação → `{approved, fraud_score}`).
- **Detecção:** vetoriza payload em **14 dims** normalizadas → **5-NN** (k=5, euclidiana) sobre
  3M vetores rotulados → `fraud_score = nº_fraudes/5`, `approved = fraud_score < 0.6`.
- **Ground truth:** rotulado com **5-NN exato float64 brute force** (AVALIACAO.md). Cada
  payload de teste tem `expected_approved`. **0 failures = reproduzir o 5-NN exato.**
- **failures = FP + FN + Err** (não é só timeout!). Nosso `Err=0` → toda failure é classificação.
- **Scoring (AVALIACAO.md):**
  - `p99_score = 1000·log10(1000/max(p99,1))` — cap **+3000** em ≤1ms, corte **−3000** acima de 2000ms.
  - `detection_score = 1000·log10(1/max(ε,0.001)) − 300·log10(1+E)`, `E = 1·FP+3·FN+5·Err`,
    `ε = E/N`; corte **−3000** se `failures/N > 15%`. Teto **+3000** (E=0).
  - `final = p99_score + detection_score`.
- **KNN NÃO é obrigatório** (FAQ): qualquer classificador vale. Líderes (p99 0.37ms / 0%) usam
  modelo treinado (ex.: xgboost), não varredura.

## Restrições (o container é apertado)

- **Hardware de avaliação:** Mac Mini Late 2014 — **2 cores, 2.6 GHz** (Haswell, tem AVX2).
- **Soma de TODOS os serviços ≤ 1 CPU e 350 MB.** Hoje: api1/api2 = **0.45 CPU + 155 MB** cada,
  nginx = 0.10 + 40 MB. `GOMAXPROCS=1`, `GOMEMLIMIT=150MiB`, `GOGC=off`.
- **Orçamento por query:** ~0.45 core a 2.6 GHz → **~1 ms de CPU/query**. Mais recall (mais
  varredura) = mais p99. É a tensão central de qualquer melhoria.
- **Memória:** live set ~96 MB (uint16) → ~50 MB livres. `float32` (168 MB) **não cabe**.
- Binário estático (`CGO_ENABLED=0`, distroless). Sem dependências externas.

## Como a API funciona

1. `internal/vectorize` — payload → `[14]float64` (normalizado [0,1], sentinela −1 nas dims
   5/6 quando `last_transaction` é null). **Vetorização é fiel ao gabarito (99.996%).**
2. `internal/knn` — quantiza cada dim pra **uint16** ([0,1]→[1,65535]; −1→0), busca os 5 mais
   próximos. Três modos via `Build`: `brute` (exato, fallback N<2048), `vptree` (exato, ensino),
   **`ivf`** (aproximado k-means, **produção**). Distância euclidiana² em acumulador uint64.
3. `internal/dataset` — carrega `references.json.gz` (3M vetores).
4. Índice IVF **pré-construído no build** (`cmd/buildindex` → `index.bin` embutido na imagem);
   no startup só carrega (~25ms). `main.go` carrega de `INDEX_PATH`, com fallback pra build no startup.

## Estado atual / jornada (docs/performance/README.md tem a tabela completa)

| etapa | técnica | p99 | failures | final | ambiente |
|---|---|---|---|---|---|
| baseline | brute-force | 2002ms | 100% | −6000 | docker |
| 03-05 | IVF + tuning + índice no build | 484ms | 0.50% | 1531 | docker (local) |
| **rank oficial** | uint8 (Mac Mini) | 562ms | 0.503% | **1460** | Mac Mini |
| **07 (atual)** | **quantização uint16** | 387ms | 0.33% | **1843** | docker (local) |

`detection_score` é **portável** (independe de HW): +1216 → **+1430** com uint16. p99 varia por
máquina (local é otimista vs Mac Mini).

## Aprendizados-chave (NÃO re-descobrir)

- **As failures eram QUANTIZAÇÃO, não recall do IVF** (provado por `cmd/diag`, doc 07): busca
  exata float64 = 0 failures; uint8 exato já errava 234; IVF só somava 35. uint8→uint16 cortou
  pra 180.
- **uint16 é o teto de precisão que cabe.** 65535 buckets ainda flipam ~144 empates de fronteira;
  só float zera, e float32 estoura a memória. Tunar `nprobe` não passa do piso de quantização.
- **VP-tree é mais LENTO que brute** em 14-D (maldição da dimensionalidade) — mantido só pra ensino.
- **Early-abandon por-dimensão não pagou** (loop de 14 dims curto demais; quebra pipeline) — revertido.
- **Caminho pro topo (0 failures + p99 sub-ms): classificador treinado** (GBDT/árvores, Go puro,
  offline) — contorna o tradeoff precisão↔memória do KNN. Próximo grande passo se mirar 6000.

## Arquivos-chave

- `main.go` — servidor; carrega `INDEX_PATH` ou faz fallback build.
- `internal/knn/{knn,ivf,vptree,serialize}.go` — busca + quantização uint16 + índice serializado v2.
- `cmd/buildindex/main.go` — builda o `index.bin` offline (no Docker build, CPU cheia).
- `cmd/diag/main.go` — **harness de diagnóstico offline** (replay do test-data, atribui failures).
- `Dockerfile` — multi-stage; builda índice no build, embute `index.bin` (sem o `.gz` no runtime).
- `docker-compose.yml` — stack de dev (build local). `submission/` — arquivos da branch submission.
- `docs/performance/` — trilha didática (00-07). `TESTING.md` — runbook de carga.

## Comandos (Go em `C:\Program Files\Go\bin`, NÃO está no PATH)

```powershell
$env:Path = "C:\Program Files\Go\bin;" + $env:Path
go test ./...                       # testes (todos verdes)
go run ./cmd/diag                   # diagnóstico offline (~8min; precisa do .gz + test-data.json)
.\run-test.ps1 -SkipSmoke           # builda imagem local + sobe stack + k6 + nota (docker)
.\verify-submission.ps1 -SkipSmoke  # dry-run da SUBMISSÃO: pull da imagem pública + k6
.\publish.ps1 -User srwalkerb       # builda uint16 + push pra Docker Hub (tag 1.0)
```

## Variáveis de ambiente

| Var | Default | Onde |
|---|---|---|
| `INDEX_PATH` | (vazio) | runtime — carrega índice pronto; senão fallback build |
| `KNN_NPROBE` | 8 (compose: 12) | runtime — células IVF varridas/query (retuna sem rebuild) |
| `KNN_NLIST` | 4096 (Dockerfile ARG) | build — nº de células k-means |
| `GOMAXPROCS/GOMEMLIMIT/GOGC` | 1 / 150MiB / off | compose |
| `ADDR` | :8080 | porta da API (nginx expõe 9999) |
| `PPROF_ADDR` | (vazio) | debug — NUNCA setar na submissão |

## Submissão (fluxo)

- **2 branches:** `main` (código-fonte completo) + `submission` (só `docker-compose.yml` por
  IMAGEM pública + `nginx.conf` + `info.json`, **sem fonte**).
- A engine puxa do GitHub (branch submission) + Docker Hub (imagem). Repo deve ser **público**.
- **Imagem precisa estar publicada e atualizada.** Tag é mutável: sobrescrever `1.0` mantém a
  branch sem mudança, MAS `docker compose` por padrão (`pull_policy: missing`) pode reusar cache
  → considerar `pull_policy: always` no compose da submission pra garantir pull fresco.
- Teste de prévia: abrir issue em `zanfranceschi/rinha-de-backend-2026` com `rinha/test` na
  descrição. Teste final: automático, **deadline 2026-06-05**.
- **PENDENTE:** a imagem publicada ainda é **uint8 antiga** — re-pushar uint16 antes do teste final.

## Ambiente de dev (quirks)

- Go fora do PATH (acima). Working dir pode driftar (use `Set-Location` pro dir da API).
- `go run .` deixa `rinha-fraud.exe` órfão segurando porta → `Get-Process rinha-fraud | Stop-Process`.
- Porta 8080 do host tomada por `ors-app` → runs locais usam :8088.
- Dataset `references.json.gz` (~48MB) e `index.bin` são gitignorados; o `.gz` precisa estar em
  `resources/` no momento do `docker build` (o buildindex consome).

## Próximo passo

1. Re-pushar a imagem **uint16** (`publish.ps1`) — a live ainda é uint8.
2. (Opcional, rumo ao topo) **classificador treinado** pra 0 failures + p99 sub-ms.
