# 08 - Quantização na grade nativa (×10000) + distância float64

> **Pré-requisito:** doc 07 (failures = quantização, não IVF; uint16 ×65534 → 180 failures / det +1429).
> Status: **✅ implementado + medido offline** (2026-05-31). Branch `feat/quantize-lossless-f64`.

## Pergunta

A doc 07 concluiu "uint16 é o teto de precisão que cabe" e que só `float64` (que estoura a
memória) zera as failures. **Isso estava certo?** Ou dava pra ter o comportamento do float64-exato
**sem** pagar a memória do float64?

## A hipótese que vira a mesa (de novo)

Dois fatos que a doc 07 não conectou:

1. **Os vetores de referência são exatamente 4 casas decimais** (verificado no `references.json.gz`:
   `0.0109`, `0.3913`, `0.8261`, … nunca >4 casas). Vivem na grade `{0, 0.0001, …, 1.0}` (10001
   valores) + sentinela −1.
2. A `quantize` antiga usava `round(v*65534)+1` — **mais buckets, porém desalinhados** com essa
   grade. E quantizava **a query** (precisão cheia no runtime) junto.

O erro dominante nos 144 flips do brute **não era o bucket da referência** — era a **quantização
da query** (~8e-6/dim, da ordem dos empates <1e-5). Prova na própria doc 07: uint8→uint16 (256×
mais buckets) só caiu 234→144. Se fosse contagem de bucket, teria caído ~256×.

**Fix:** guardar a referência como `round(v*10000)+1` em uint16 (**lossless** p/ dado 4-casas; cabe
em 84MB) e computar a distância em **float64**, dequantizando a ref (`(u-1)/10000`) contra a query
**crua, não quantizada**. Algebricamente é o `bruteF64` do `cmd/diag` (o oráculo). Sentinela −1 → 0.

## Implementação

`internal/knn`: `quantize` ×10000 + helper `dequant`; `dist2`/`rowDist2`/`topK.dist`/`centroidDist`
viraram **float64** dequantizando inline; `Score` deixou de quantizar a query (passa o `[14]float64`
direto); centróides do IVF agora `[]float64` (k-means sem arredondar); `serialize` bump **v2→v3**
(+`writeF64Slice`/`readF64Slice`). `index.bin` rebuildado (92.4 MB). `go test ./...` verde
(`TestVPTreeExactMatchesBrute` = VP exato == brute; `TestDequantRoundTrip` = `dequant(quantize(v))==v`
bit-exato p/ 4-casas; `TestIVFRoundTrip` v3).

## Resultado (`cmd/diag`, 54100 entries, 2026-05-31)

| Método | FP | FN | failures | fail% | det_score | (era ×65534) |
|---|---|---|---|---|---|---|
| float64-exact (oráculo) | 0 | 0 | **0** | 0.000% | **+3000** | — |
| **u16/f64-brute (×10000)** | 0 | 0 | **0** | 0.000% | **+3000** | 144 / +1532 |
| **IVF (4096/12)** | 25 | 9 | **34** | 0.063% | **+2482.7** | 180 / +1429 |

**Atribuição das 34 failures do IVF:** vetorização 0% · quantização **0% (era 80%)** · **recall 100%
(34)**. Concentração na fronteira: 21 em 2/5, 9 em 3/5, 4 em 1/5. Fidelidade da vetorização: 99.996%
(inalterada).

**Veredito:** o piso de quantização (144→**0**) sumiu. O brute agora reproduz o gabarito **exatamente**
(det **+3000**) dentro de 84MB. As failures restantes são **só recall do IVF** — exatamente o que a
doc 07 previu que aconteceria se a quantização fosse zerada. `detection_score` do que SHIPa (IVF) subiu
**+1429 → +2482.7** (Δ **+1053**), e isso é **portável** (independe do Mac Mini).

## Sweep IVF ×10000 (recall vs custo) — a alavanca que sobrou

| config | failures | fail% | det_score | linhas/query |
|---|---|---|---|---|
| nlist=4096 nprobe=8 | 44 | 0.081% | +2316.9 | 5.856 |
| nlist=4096 nprobe=12 | 34 | 0.063% | +2482.7 | 8.784 |
| nlist=4096 nprobe=16 | 24 | 0.044% | +2529.5 | 11.712 |
| nlist=4096 nprobe=24 | 14 | 0.026% | +2580.6 | 17.568 |
| nlist=4096 nprobe=32 | 7 | 0.013% | +2676.2 | 23.424 |
| nlist=8192 nprobe=12 | 32 | 0.059% | +2458.2 | 4.392 |
| nlist=8192 nprobe=24 | 17 | 0.031% | +2575.5 | 8.784 |

Agora subir nprobe **vale a pena** (antes o piso era 234/quantização): de nprobe=12 a 32, failures
34→7 e det +2482→+2676 — mas custa ~2.7× linhas/query (p99). O ponto ótimo depende do que o p99 paga
no Mac Mini. `nlist=8192 nprobe=24` dá det parecido com `4096/24` pela metade das linhas/query — boa
candidata. **Medir no docker (`run-test.ps1`) antes de fixar.**

## Resultado sob carga (docker, local, 2026-05-31)

Medido numa **stack isolada** (build local da imagem com o código novo + index v3; rede própria;
caps mirror do eval: api 0.45/155MB ×2, nginx 0.10/40MB; k6 na netns do nginx). `nprobe=12`.

| versão | p99 | failure_rate | p99_score | detection_score | **final_score** | Err |
|---|---|---|---|---|---|---|
| dequant com divisão `/10000` | 702.75ms | 0.06% | 153.2 | 2498.4 | **2651.6** | 0 |
| **dequant via LUT (mul-free)** | **566.54ms** | 0.06% | 246.8 | 2495.6 | **2742.4** | 0 |

- `final_score` **1843 → 2742** (Δ **+899**). `detection_score` **1429 → 2496** (Δ +1067) — **portável**
  (independe de HW; vale no Mac Mini). `Err=0` (100% atendido @ 900 req/s).
- **Pegadinha de p99 encontrada e corrigida:** a `dequant` fazia uma **divisão por dimensão**
  (`/10000`) no laço quente (~123k divisões/query). Trocada por **lookup table precomputada**
  (`dequantTab`, bit-exata) → p99 702→566ms, final +91. Lição: float é ok, **divisão no hot path não**.
- p99 (566ms local) ainda inflado por **contenção do host** (10 containers competindo na medição);
  número de detecção é o ganho sólido/portável. p99 limpo tende a ficar bem menor.

## Próximo passo

- (Opcional) sweep `nprobe`/`nlist` sob carga pra achar o ponto p99↔detecção (diag mostrou
  `nlist=8192 nprobe=24` como candidata: det ~+2575 com metade das linhas/query de `4096/24`).
- **Caminho C (GBDT)** ataca o p99 (gap dominante: 247 vs teto 3000) — agora com os números do A
  medidos na mão. É o próximo plano.
