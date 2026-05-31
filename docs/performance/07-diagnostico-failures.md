# 07 - Diagnóstico das failures (por que erra 0.5%)

> **Pré-requisito:** estar no ranking (#166, p99 562ms, **failures 0.503%**, score 1460).
> Status: **✅ diagnóstico feito** (2026-05-31). Ferramenta: `cmd/diag` (offline, não toca produção).

## Pergunta

As "failures" do ranking — **o que são** e **por que acontecem**? E como chegar a 0?

## O que o desafio define (AVALIACAO.md)

- `failures = FP + FN + Err`. Nosso `Err = 0` → **toda failure é classificação errada**, não
  timeout. A API responde tudo; só dá o `approved` errado em ~0.5%.
- Ground truth = **5-NN exato float64 brute force** sobre as referências. Cada entry de
  `test-data.json` (54.100 rotulados) traz `expected_approved` + `expected_fraud_score`.
- `detection_score = 1000·log10(1/max(ε,0.001)) − 300·log10(1+E)`, `E = 1·FP+3·FN+5·Err`.
  Hoje E≈501 → det ≈ +1212. Teto (E=0) = **+3000**.

## Como medimos (`cmd/diag`)

Reproduz a avaliação **localmente** (sem docker/k6): vetoriza cada request e roda **três**
métodos sobre os 3M de referências, comparando com `expected_approved`:
1. **float64-exato** (nossa `Vectorize` + refs cru) → mede fidelidade da vetorização.
2. **uint8-brute** (busca exata sobre dados quantizados) → mede custo da quantização.
3. **IVF (4096/12)** (o que está em produção) → mede custo do recall.

Atribui cada failure pontuada (IVF≠expected) a uma causa: vetorização → quantização → recall.

## Resultado (2026-05-31, 54.100 entries) — A SURPRESA

| Método | FP | FN | failures | fail% | det_score |
|---|---|---|---|---|---|
| **float64-exato** | 0 | 0 | **0** | 0.000% | **+3000** |
| uint8-brute (exato) | 120 | 114 | 234 | 0.433% | +1269 |
| IVF 4096/12 (produção) | 148 | 121 | 269 | 0.497% | +1212 |

**Atribuição das 269 failures pontuadas:**
- vetorização: **0 (0%)** — nossa `Vectorize` é perfeita (fraud_score bate em 99.996%).
- **quantização uint8: 234 (87%)** ← a causa dominante.
- recall do IVF: 35 (13%).

**Conclusão que vira a mesa:** a culpa **não é do IVF** (como eu assumi na trilha). É a
**quantização para uint8** (255 buckets por dimensão). A busca exata float64 acerta **100%**
(`det +3000`); a busca exata uint8 já erra 234. O IVF só adiciona 35.

**Onde as failures moram:** 236 das 269 (88%) estão exatamente na fronteira (2/5 ou 3/5
fraudes) — a quantização empurra um vizinho borderline pra dentro/fora e vira a maioria
através da linha 0.6.

**Sweep IVF (prova que tunar IVF é beco sem saída):**

| config | failures | fail% | det_score | linhas/query |
|---|---|---|---|---|
| nlist=4096 nprobe=8 | 274 | 0.506% | +1209 | 5.856 |
| nlist=4096 nprobe=12 | 269 | 0.497% | +1212 | 8.784 |
| nlist=4096 nprobe=16 | 267 | 0.494% | +1216 | 11.712 |
| nlist=4096 nprobe=24 | 257 | 0.475% | +1235 | 17.568 |
| nlist=4096 nprobe=32 | 248 | 0.458% | +1250 | 23.424 |
| nlist=8192 nprobe=24 | 252 | 0.466% | +1227 | 8.784 |

Mesmo a 4× o custo (nprobe=32), as failures só caem de 269→248 — porque o **piso é a
quantização (234)**. Recall do IVF já está colado nesse piso. Subir nprobe gasta p99 e **não
fecha** as failures.

## Veredito + menu de fix (decidido com dados)

A vetorização é perfeita e a busca em alta precisão zera as failures (det +3000). Logo o alvo
é a **precisão dos dados**, não o IVF. Avaliado contra o orçamento do container (≤~1 ms
CPU/query a 2.6 GHz, ~100 MB livres, soma ≤ 1 CPU / 350 MB):

- **❌ Tunar IVF (nprobe/nlist):** não passa do piso de 234 e custa p99. Descartado pra failures.
- **❌ float64 (precisão total):** zera failures, mas 3M×14×8 = **336 MB** > limite de 155 MB. Não cabe.
- **✅ FEITO — quantizar para `uint16` em vez de `uint8`:** 3M×14×2 = **84 MB** (cabe folgado no
  GOMEMLIMIT 150MiB). Mais precisão → menos flips de fronteira. Custo de CPU/p99 quase nulo.

## Resultado do fix (uint16, 2026-05-31)

Implementado (quantize/data/dist2/serialize v2). Re-rodando o `cmd/diag` com uint16:

| método | failures | fail% | det_score |
|---|---|---|---|
| float64-exato | 0 | 0.000% | +3000 |
| **u16-brute (exato)** | **144** | **0.266%** | **+1532** |
| **IVF u16 (4096/12)** | **180** | **0.333%** | **+1429** |

- failures `269 → 180` (−33%); det_score `+1212 → +1429` (**+217**). Atribuição: quantização
  ainda 80% (144), recall 20% (36), vetorização 0%.
- **Não chegou a ~0 (minha projeção estava otimista).** Lição honesta: mesmo **65.535 buckets**
  ainda trocam **empates genuínos** na fronteira (5º vs 6º vizinho colados a <1e-5). Só
  precisão float (float64 = 0 failures) elimina — e `float32` (168 MB) **estoura** o limite de
  155 MB/instância. Então **uint16 é o teto de precisão que cabe** no container.

**Sweep uint16 (recall vs custo):** nprobe 12→32 baixa 180→152 mas custa ~2.7× linhas/query
(p99 sobe no Mac Mini) — provavelmente não compensa (perde mais p99_score do que ganha em
detecção). nprobe=12 segue o ponto de equilíbrio. Medir no docker confirma.

## Veredito final

- **uint16 é ganho líquido e seguro → enviar.** Esperado `final ~1460 → ~1600-1700` (det +217;
  p99 ~igual, talvez leve alta pela banda 2×).
- **Para chegar a ~0 failures E p99 sub-ms** (topo do ranking) o caminho é o **classificador
  treinado** (árvores/GBDT offline em Go puro): cabe em <1 MB, infere sub-ms, e pode reproduzir
  o 5-NN float64 sem o tradeoff precisão↔memória do KNN. É o próximo grande passo, se quiser
  mirar os 6000.

## Próximo passo

Rebuildar a imagem (buildindex gera `index.bin` v2 uint16) e medir o `final_score` real com
`run-test.ps1` sob os limites do container.
