# Trilha de Performance — rinha-fraud

Aprender otimização de performance de API **aplicando uma técnica por vez**, medindo o
impacto de cada uma. Cada doc é autocontido: lê, aplica no código, mede, registra o
resultado na tabela abaixo, e só então segue pro próximo.

> **Princípio-mestre:** *meça antes de complicar.* Nunca otimize no escuro — primeiro prove
> onde está o gargalo, depois mexa, depois prove que melhorou.

## Por que esta trilha existe

O baseline mede `final_score = -6000` (a pior nota possível) sob carga: a busca
força-bruta sobre 3 milhões de vetores não aguenta 900 requisições/segundo e **tudo dá
timeout**. A meta é cruzar os dois "cortes" do scoring (latência e taxa de falha) e empurrar
o `final_score` pro positivo — entendendo cada passo.

## A trilha

| # | Técnica | O que ensina | Status |
|---|---------|--------------|--------|
| [00](00-metodologia-e-medicao.md) | Metodologia e medição | Teste de carga (k6), fórmula de score, intro a profiling (pprof) | ✅ |
| [01](01-perfil-do-gargalo.md) | Perfilar o gargalo | Ligar o pprof, capturar CPU profile, **provar** o gargalo | ✅ |
| [02](02-ganhos-baratos.md) | Ganhos baratos | Benchmark do Go; early-abandon (não pagou → revertido); por que JSON é ruído | ✅ |
| [03](03-busca-rapida.md) | Busca rápida | Trocar O(3M) por índice espacial: VP-Tree (exato) + IVF (aproximado) | ✅ |
| [04](04-tuning-runtime.md) | Tuning de runtime | `GOMAXPROCS`, `GOGC=off`, `GOMEMLIMIT`, rebalance de CPU | ✅ |
| [05](05-preprocessamento-no-build.md) | Pré-processo no build | Construir o índice no Docker build (startup rápido, nlist grande) | ✅ (p99 1202→484ms, +395) |
| [06](06-simd-avancado.md) | SIMD (avançado) | Vetorizar o cálculo de distância | 📋 guia (opcional, `GOAMD64=v3` não ligado) |
| [07](07-diagnostico-failures.md) | Diagnóstico das failures | Harness offline; failures = quantização (não IVF!); fix uint16 | ✅ (det +1216→+1430) |

Template para criar/seguir cada doc: [`_template.md`](_template.md).

## Tabela de progresso

A fonte única da verdade do que cada técnica rendeu. Uma linha por medição. Valores copiados
direto do `results.json` gerado pelo `run-test.ps1`.

> ⚠️ **Coluna `ambiente` é sagrada.** Só compare linhas do **mesmo** ambiente:
> - `docker-9999` — stack real (2 instâncias com cap de 0.40 CPU atrás do nginx). **É a nota que vale.**
> - `local-8080` — `go run .` sem cap de CPU (rápido pra iterar, números otimistas).
> - `local-pprof` — rodando com profiling ligado (números distorcidos, só pra achar hotspot).

| etapa | técnica | data | p99 | failure_rate | p99_score | detection_score | final_score | Δ final | ambiente | notas |
|-------|---------|------|-----|--------------|-----------|-----------------|-------------|---------|----------|-------|
| baseline | brute-force | 2026-05-30 | 2002ms | 100% | -3000 | -3000 | **-6000** | — | docker-9999 | colapso de throughput (32 rps de capacidade vs 900 de demanda) |
| 03 | IVF nlist=1024 nprobe=12, cpu 0.40 | 2026-05-30 | 2002ms | 7.86% | -3000 | -751 | **-3751** | +2249 | docker-9999 | detecção saiu do corte (FP/FN minúsculos); p99 ainda no corte |
| 04 | + nprobe=8, cpu 0.45/nginx 0.10 | 2026-05-30 | 1515ms | 0.63% | -180 | +959 | **+778** | +4529 | docker-9999 | p99 saiu do corte; nota POSITIVA; 45 timeouts |
| 04 | + nprobe=6, GOGC=off | 2026-05-30 | 1202ms | 0.51% | -80 | +1217 | **+1136** | +358 | docker-9999 | **Err: 0** — 100% das requests atendidas (40.300, zero timeout) |
| 05 | índice no build, nlist=4096 nprobe=12 | 2026-05-30 | 484ms | 0.50% | +315 | +1216 | **+1531** | +395 | docker-9999 | startup 25ms (era ~85s); **p99 1202→484ms**; Err=0; o 0.5% é FP+FN (classificação), não timeout |
| 07 | quantização **uint16** (era uint8) | 2026-05-31 | 387ms | 0.33% | +413 | +1430 | **+1843** | +312 | docker-9999 | diagnóstico provou: failures eram QUANTIZAÇÃO, não IVF. uint16 → det +1216→**+1430** (portável). p99 não piorou (banda 2× irrelevante; IVF varre poucas linhas) |

**Estado atual (local): `+1843`** (rank oficial no Mac Mini foi 1460 com uint8). Objetivo "0
timeout / 100% atendido" ✅ (`Err=0` desde o passo 04). A jornada: p99 caiu de 2002ms→**387ms**
(passos 03-05) e as failures de classificação de 0.5%→**0.33%** (passo 07, uint16). O ganho do
passo 07 é o **`detection_score` +1216→+1430**, que é **portável** (independe de hardware) — então
vale igual no Mac Mini. As failures restantes (0.33%) são FP+FN de fronteira; o piso de precisão
do uint16 é ~144 (doc 07). Próximo salto de verdade (rumo a 0 failures + p99 sub-ms) = **classificador
treinado** (doc 07, veredito final).

## Como medir (atalho)

```powershell
cd C:\Me\me-projects\rinha-backend-2026\rinha-de-backend-2026-api
.\run-test.ps1            # sobe a stack + k6 carga + imprime p99/failure_rate/final_score
```

Detalhes e flags em [`../../TESTING.md`](../../TESTING.md).
