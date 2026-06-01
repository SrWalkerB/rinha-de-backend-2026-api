# Design — Quantização lossless + classificador GBDT (rinha-fraud)

**Data:** 2026-05-31 · **Status:** aprovado, pré-implementação · **Deadline competição:** 2026-06-05

## Contexto

A API de detecção de fraude (`rinha-de-backend-2026-api`, Go puro) hoje classifica via 5-NN
sobre 3M vetores 14-dim, índice IVF, quantização uint16. Estado medido: `final_score` **1843**
local / **~1680 Mac Mini** (uint16 ainda não publicado). Submissão atual é válida e **não pode
quebrar** (`Err=0`, 100% atendido).

Dois muros separam dos ~6000 do topo:

1. **Detecção** — failures ≈ 0.33% (180 no IVF / 144 no brute, uint16). `cmd/diag` provou (doc
   `docs/performance/07`) que **5-NN float64 exato = 0 failures** (é a definição do gabarito); a
   culpa é a **quantização**, não o IVF nem a vetorização (99.996% fiel).
2. **p99** — ~387ms local / ~562ms Mac Mini. `p99_score` ≈ +413 vs teto +3000. Gap dominante
   (~2587 pts). A varredura KNN (~180k ops/query) não chega a sub-ms sob `cpus:0.45`.

`p99_score = 1000·log10(1000/max(p99,1))` (teto +3000 em ≤1ms); `detection_score =
1000·log10(1/max(ε,0.001)) − 300·log10(1+E)`, `E=1·FP+3·FN+5·Err` (teto +3000 em E=0).
`final = p99_score + detection_score`.

## Achado que orienta o design

Os vetores de referência (`references.json.gz`) são **exatamente 4 casas decimais** (verificado:
0.0109, 0.3913, 0.8261, … nunca >4 casas). Logo vivem na grade `{0, 0.0001, …, 1.0}` (10001
valores) + sentinela −1. Essa grade **cabe lossless em uint16** via `round(v*10000)`.

A quantização atual usa `round(v*65534)+1` — mais buckets, porém **desalinhados** com a grade
4-casas: reconstrói `0.3913 → 0.391307…`, erro ~1e-5/dim, **da ordem dos empates de fronteira**
(<1e-5), o que vira os 144 flips. O número de buckets nunca foi o gargalo; o **desalinhamento**
e a quantização da query foram.

## Decisão

Implementar **dois scorers independentes**, selecionáveis por env, medidos lado a lado; **subir
na submissão o que pontuar mais**. Mantém os dois no código. KNN melhorado serve de rede de
segurança forte se o modelo decepcionar.

```
SCORER=knn  (Caminho A — não-IA)            SCORER=model  (Caminho C — IA)
 vectorize → ref uint16 ×10000 (lossless)     vectorize → model.Score (árvores GBDT)
           → dist FLOAT64 vs query crua                  → P(fraude) → approved = P<τ
           → IVF/brute → 5-NN → fraud_score    Go puro, <1MB, sub-ms, sem varredura
 det: brute ~0 / IVF ~dezenas (recall)        det: provável <180, 0 NÃO garantido
 p99: ~igual (varre)                          p99: SUB-MS
 est. final ~2700–3000                         est. final ~4000–6000
```

### Por que não é cumulativo num motor só

A (quantização) melhora o **caminho KNN**. C (modelo) é um **caminho novo** que nem quantiza
nem varre. São dois experimentos paralelos; medir e promover o melhor.

## Caminho A — quantização lossless + distância float (não-IA)

**Ideia:** guardar a referência como inteiro `round(v*10000)` em uint16 (lossless p/ dado
4-casas) e computar distância em **float64** contra a query crua (não quantizada). Isso é
**algebricamente idêntico** ao `float64-exato` do `cmd/diag` (já medido **0 failures**), sem
pagar a memória do float64 (segue 84MB uint16; dequantiza inline).

**Contrato de armazenamento (uint16):**
- sentinela `−1` → `0`
- valor `v ∈ [0,1]` → `round(v*10000)+1` ∈ `[1, 10001]`

**Dequantização (uint16 → float64) na distância:**
- `0` → `−1.0` (sentinela)
- `s ≥ 1` → `(s−1)/10000.0`

**Distância por dimensão:** `diff = q_float[i] − dequant(ref[i]); d += diff*diff` (acumulador
**float64**). Reproduz o gabarito: para `v` 4-casas, `round(v*10000)/10000` = o mesmo double que
o JSON parseia → mesma ordenação que o float64-exato.

### Arquivos
- `internal/knn/knn.go` — `quantize` (×10000); `dist2`/`rowDist2`/`topK.dist` → float64
  dequantizando inline. `Threshold` (0.6) inalterado.
- `internal/knn/ivf.go` — `centroidDist` → float64; centróides guardados como `[]float64`
  (k-means já produz médias float; sem arredondar pra uint16). Memória dos centróides
  desprezível (nlist×14×8 ≈ 458KB p/ nlist=4096).
- `internal/knn/serialize.go` — bump **v2 → v3** (escala + tipo de centróide mudaram; `index.bin`
  antigo incompatível). Ler/escrever centróides float64.
- `cmd/buildindex` — sem mudança lógica; só rebuilda com o novo `quantize`.
- testes — `quantize(1)==10001`, `quantize(-1)==0`, roundtrip serialize v3, igualdade de score
  brute vs IVF onde aplicável.

### Medição (portão A)
`cmd/diag` (sem código novo — os métodos `u16-brute` e `IVF` passam a usar ×10000): esperado
brute ≈ **0**, IVF ≈ dezenas (recall). Depois `run-test.ps1` no docker p/ p99/final reais.

### Risco
Baixo, localizado. Mudar de inteiro→float na distância: p99 ~igual (mul float64 ≈ int em
volume; IVF varre poucas linhas). Caso o brute não dê ~0, o `cmd/diag` mostra antes de subir.

## Caminho C — classificador GBDT (IA)

### Treino (offline, Python — único lugar com Python; não entra no runtime nem na imagem)
- `train/train.py`: lê `references.json.gz` (campos `vector`[14] + `label`, **já vetorizados** —
  Python NÃO replica `vectorize.go`). Treina **xgboost** (`binary:logistic`; ~200 árvores,
  `max_depth` 5–6 — ajustar medindo).
- **Threshold τ:** xgboost dá `P(fraude) ∈ [0,1]`. Calibra τ num **split de validação tirado das
  referências** (NÃO do `test-data.json` = sem vazamento) maximizando concordância com o rótulo
  5-NN. O scoring compara só `approved`, então τ é o que importa.
- Exporta `resources/model.json` via `booster.dump_model(dump_format='json')` (árvores: `split`
  feature, `split_condition`, `yes`/`no`/`missing`, `leaf`) + metadados (`base_score`, `τ`).
- `model.json` é **commitado** (poucas centenas de KB). Rodar treino é passo manual; regenerar ao
  re-treinar.

### Inferência (Go puro, runtime)
- `internal/model/model.go`: `LoadModel([]byte) (*Model, error)`; `Score([14]float64) float64`
  (= `P(fraude)`): pra cada árvore anda da raiz (`x[feature] < split_condition → yes senão no`),
  soma folhas → margin; `P = sigmoid(margin + base_margin)`. Sem dependências.
- `go:embed resources/model.json` no binário.

### Integração
- Interface `Scorer { Score([14]float64) float64; ApprovedThreshold() float64 }`. KNN `*Index` e
  `*model.Model` implementam.
- `main.go`: env `SCORER` (`knn` default | `model`). `handleScore` usa o `Scorer` escolhido.
  Mudança mínima no handler; fallback `{approved:true, fraud_score:0}` preservado.

### Arquivos
- `train/train.py` (novo), `train/README.md` (como rodar).
- `resources/model.json` (novo, commitado).
- `internal/model/model.go` + `model_test.go` (novo).
- `internal/knn/knn.go` ou novo `internal/scorer` — interface `Scorer`.
- `main.go` — seleção por env.
- `cmd/diag` — + método "model" na comparação.

### Teste-âncora (parity, crítico — TDD)
`internal/model` deve reproduzir o xgboost: salvar no treino um sample `(vetor → predict_proba)`;
teste Go exige `model.Score(v)` ≈ Python `predict_proba(v)` dentro de `1e-6`. Sem isso, o modelo
em Go não é confiável.

### Medição (portão C)
`cmd/diag` (método model, 54100 entries): failures vs float64-exato(0)/IVF. Depois `run-test.ps1`
docker p/ p99 (esperado sub-ms) + final.

### Risco
- xgboost pode não zerar failures (empates de fronteira) — mas dificilmente pior que IVF, e o
  ganho de p99 compensa com folga. Pior caso: mantém `SCORER=knn`.
- Parsing do dump xgboost em Go + paridade do sigmoid/base_score — coberto pelo teste de paridade.
- Setup Python na máquina de dev (Go nem está no PATH). Setup único, offline.

## Sequência

1. **A** (de-risca detecção, barato): implementa → `cmd/diag` → docker → registra.
2. **C** (lift maior): treino → inferência Go + paridade → `cmd/diag` → docker → registra.
3. **Promoção:** subir na branch `submission` o `SCORER` com maior `final_score` (bater 1843).

## Portões de segurança (invariantes)

- Cada caminho **medido no `cmd/diag` (offline) antes do docker**, e no docker antes de promover.
- Submissão só muda se `final_score > 1843`. Senão, KNN ×10000 (ou o uint16 atual) segue.
- `Err` deve seguir **0** (100% atendido) em qualquer caminho — fallback rápido preservado.

## Fora de escopo

- Projeto TypeScript de aprendizado de ML — ciclo separado (design → impl) depois, conforme
  combinado.
- Re-publicar imagem / atualizar branch `submission` — passo final, após medir e escolher o scorer.

## Ledger

A trilha `docs/performance/` (convenção do projeto, ledger de medições) ganha docs novos durante
a implementação: **08** (quantização ×10000) e **09** (GBDT), cada um com p99/failures/final
medidos por ambiente.
