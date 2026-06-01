# Jornada e decisões — de KNN a (rumo ao) GBDT

> Documento de **estudo**. Narra o processo inteiro e o **porquê** de cada decisão, pra aprender
> performance + introdução a ML na prática. Companion técnico dos docs numerados (00-08) e das
> specs/planos em `docs/superpowers/`. Linguagem informal de propósito.
>
> Sessão de 2026-05-31. Estado no fim do Caminho A: `final_score` **1843 → 2742**.

---

## 0. O mapa mental: onde os pontos moram

Tudo gira em torno da fórmula de pontuação (AVALIACAO.md). Ela tem **dois componentes
independentes**, cada um de −3000 a +3000, somados:

```
final_score = p99_score + detection_score          (faixa −6000 .. +6000)

p99_score       = 1000 · log10(1000 / max(p99, 1))   → teto +3000 em p99 ≤ 1ms
                                                       → corte −3000 em p99 > 2000ms
detection_score = 1000 · log10(1/max(ε,0.001)) − 300·log10(1+E)
                  E = 1·FP + 3·FN + 5·Err   ;   ε = E/N   ;   corte −3000 se failures/N > 15%
```

Duas leituras que guiaram TODA a estratégia:

1. **É logarítmico.** Cada **10× de melhora** no p99 vale +1000. 100ms→10ms = +1000; 10ms→1ms =
   +1000; abaixo de 1ms não rende mais nada. Ou seja: não adianta otimizar o que já está bom;
   adianta atacar ordens de grandeza.
2. **São dois muros separados.** Latência (p99) e qualidade (detecção) não se misturam. Dá pra
   ganhar num e perder no outro. Pra chegar aos 6000 do topo, precisa **dos dois** no teto.

Onde estávamos no começo da sessão (uint16, medido): p99 ~387ms → `p99_score` ~+413; failures
0.33% → `detection_score` ~+1430; **final 1843**. Gaps: p99 faltava **~2587**, detecção **~1570**.
O p99 era o gargalo maior — mas, como se verá, os dois tinham a **mesma causa raiz** e uma delas
tinha uma saída barata que a trilha tinha deixado passar.

---

## 1. A pergunta que abriu a sessão

*"Dá pra atingir 100% das requisições e os scores altos mesmo com Go?"* — olhando o ranking, a
submissão `rinha-backend-2026-go` tinha p99 0.445ms / 0% failures / **score 6000**. Logo: **sim,
Go chega ao topo.** A pergunta vira *o que está nos travando?*.

**Cuidado com ambiguidade de "100%":** no nosso caso `Err=0` desde o passo 04 (100% das requisições
são *respondidas*, sem timeout). O "100%" do ranking (`FAILURES 0.000000%`) é **0 erros de
classificação** (FP+FN+Err). São coisas diferentes — a confusão aqui é comum e vale fixar.

---

## 2. Diagnóstico do gargalo: os dois muros do KNN

A API classificava fraude por **5-NN**: vetoriza a transação em 14 dims → acha os 5 vetores de
referência mais próximos (entre 3M) → `fraud_score = nº_fraudes/5` → `approved = score < 0.6`.
Índice **IVF** (k-means particiona em células; a query varre só as `nprobe` células mais próximas).

KNN bate em **dois muros ao mesmo tempo**:

**Muro 1 — compute (mata o p99).** Por query o IVF faz ~180k operações: varre os 4096 centróides
pra escolher células (~57k ops) + varre ~8784 linhas das células escolhidas (~123k ops). Sob o cap
de `0.45` CPU por instância e 900 req/s, isso vira **centenas de ms** de p99. Não tem `nprobe` que
faça uma varredura de milhares de vetores virar sub-ms.

**Muro 2 — precisão↔memória (trava a detecção).** O gabarito é **5-NN float64 exato**. Reproduzir
exato = 0 failures. Mas:
- guardar 3M×14 **float64** = 336MB → estoura o limite de 155MB/instância;
- guardar **uint16** (84MB) cabe, mas quantizar perde precisão → empates de fronteira flipam.

A doc 07 já tinha provado (com o `cmd/diag`) que **a vetorização é fiel (99.996%)** e que as
failures eram **quantização**, não recall do IVF — e tinha concluído, pessimista: *"uint16 é o teto
de precisão que cabe; só float zera, e float não cabe."*

**Lição de método:** o `cmd/diag` (harness offline que replica a avaliação sem docker/k6) foi o que
permitiu *atribuir* cada failure a uma causa — vetorização vs quantização vs recall. Sem medir a
causa, a gente teria "otimizado no escuro" (provavelmente mexido no IVF, que não era o problema).

---

## 3. O detour de ML — e por que quase não precisou

A ideia inicial pro topo era **classificador treinado** (GBDT/árvores). Vale registrar a discussão
conceitual porque ela esclarece o que é "IA" aqui:

- **KNN já é machine learning** (método "preguiçoso": decora os dados, adia o trabalho pra query).
  Um **GBDT** é ML "aplicado": treina offline (escolhe limiares que minimizam erro nos 3M exemplos),
  joga os dados fora, e na query só **avalia uma função** (uns `if` aninhados somando folhas). Mesmo
  objetivo, trade-off oposto: o trabalho pesado sai do runtime.
- **"É IA?"** Sim, no sentido amplo (ML clássico) — mas longe de rede neural/LLM. "Treino" = um loop
  offline ajustando números. Determinístico. E o **runtime continua Go puro** (o modelo vira uma
  tabela de floats embutida no binário; inferência é aritmética). Não fere "Go puro, sem deps,
  350MB".
- Espectro: `regras na mão → KNN (estávamos aqui) → árvore/GBDT → rede neural → LLM`. A gente sobe
  **um** degrau, não vai pro fundo da piscina.

**A virada:** antes de partir pra IA, perguntei *"isso é mesmo o melhor caminho? tem alternativa?"*.
E aí o achado decisivo: **os vetores de referência são exatamente 4 casas decimais** (`0.3913`,
`0.0833`, … nunca >4). Isso muda o jogo do Muro 2 — e **não precisa de IA pra detecção**.

---

## 4. As alternativas (e por que cada uma)

Pro **Muro 2 (detecção / 0 failures)**:

| alternativa | veredito | porquê |
|---|---|---|
| tunar `nprobe`/`nlist` do IVF | ❌ | doc 07 provou: piso era quantização (234/uint8, 144/uint16), não recall |
| float64 exato | ✅ zera, ❌ cabe | 336MB > 155MB/instância |
| float32 | ❌ | 168MB ainda estoura |
| **uint16 na grade nativa ×10000 + distância float64** | ✅✅ | **dado 4-casas cabe lossless em uint16; distância em float reproduz o gabarito** |
| GBDT (IA) | ~ | aproxima a fronteira; **não garante 0**; e é a alavanca do Muro 1, não do 2 |

Pro **Muro 1 (p99 sub-ms)**: só **parar de varrer** resolve. KNN exato (VP-tree) é mais lento em
14-D (maldição da dimensionalidade). Então o p99 sub-ms **exige** um classificador treinado (ou
estrutura O(1) equivalente). É aí que o GBDT (Caminho C) entra — não pela detecção, pelo p99.

**Conclusão estratégica:** dois muros, alavancas diferentes. Detecção → fix barato não-IA (Caminho
A). p99 → GBDT (Caminho C). Fazer A primeiro (barato, de-risca) e C depois.

---

## 5. A correção do meu próprio raciocínio (importante pro estudo)

Primeira ideia: *"quantizar tudo ×10000, inteiro"*. **Erro.** Isso arredondaria também a **query**
(que no runtime é precisão cheia: `hora/23 = 0.21739…`). E foi exatamente a **quantização da query**
(não dos buckets da referência) que dominava os 144 flips. Prova nos próprios números da doc 07:
uint8→uint16 = **256× mais buckets**, mas failures só caíram 234→144 (1.6×). Se fosse contagem de
bucket, teria caído ~256×. Logo o erro não escala com bucket → é a query.

Versão certa (Caminho A): guardar a **referência** lossless em uint16 (×10000) e calcular a
distância em **float64** contra a query **crua, não quantizada**:

```
dist² = Σ ( query_float[i] − dequant(ref_uint16[i]) )²      // tudo em float64
dequant(0) = −1 (sentinela) ;  dequant(k) = (k−1)/10000
```

Como o dado é 4-casas, `(k−1)/10000` reproduz **bit a bit** o mesmo `float64` que o JSON parseia →
a distância é idêntica à do gabarito (float64 exato). Memória: continua uint16 (84MB).

**Lição:** distinguir *qual lado* da conta perde precisão. A referência era recuperável; a query
não — então a query **não pode** ser quantizada se a meta é zero. (E "mais buckets" não era a
resposta; "alinhar à grade do dado + não quantizar a query" era.)

---

## 6. Caminho A — implementação (não-IA)

O que mudou em `internal/knn` (migração de distância **inteiro→float64**, atômica — o pacote não
compila no meio, então mudou tudo junto e depois compilou):

- `quantize`: `round(v*65534)+1` → `round(v*10000)+1` (grade nativa, lossless p/ 4-casas).
- novo `dequant` (uint16 → float64 exato), e `dist2`/`rowDist2`/`topK.dist`/`centroidDist` viraram
  **float64** dequantizando a referência inline.
- `Score` **parou de quantizar a query** (passa o `[14]float64` direto).
- centróides do IVF agora `[]float64` (k-means sem arredondar).
- `serialize` v2→v3 (formato incompatível; `index.bin` rebuildado).

Feito com **TDD**: primeiro o teste (`quantize(1)==10001`, `dequant(quantize(v))==v` bit-exato),
vê falhar, implementa, vê passar. `TestVPTreeExactMatchesBrute` (VP exato == brute) e
`TestIVFRoundTrip` (serialize v3 round-trip) seguraram a migração.

### A pegadinha de p99 (ótima lição de performance)

Primeira medição no docker: detecção ótima, mas **p99 piorou (702ms vs 387ms)**. Causa: o `dequant`
fazia uma **divisão** (`/10000`) **por dimensão, no laço quente** (~123k divisões/query). Divisão
custa ~20-40 ciclos vs ~1 do mul. Fix: **lookup table precomputada** (`dequantTab`, 80KB, calculada
uma vez com `/10000` → bit-exata) → o laço faz um *lookup* em vez de dividir. p99 702→**567ms**,
final +91.

**Lição:** `float64` no hot path é ok (mul/sub são baratos); **divisão não é**. Quando precisa de
uma transformação fixa por elemento, **precompute numa tabela**.

---

## 7. Resultados medidos

`cmd/diag` (offline, 54100 entries — o conjunto que a engine usa):

| método | failures | det_score | (era ×65534) |
|---|---|---|---|
| float64-exact (oráculo) | **0** | +3000 | — |
| brute ×10000 + float64 | **0** | +3000 | 144 / +1532 |
| IVF 4096/12 | **34** (só recall) | **+2482.7** | 180 / +1429 |

Docker (stack isolada, caps espelhando o eval, host contido):

| | antes | depois |
|---|---|---|
| p99 | — | 567ms (host contido; número limpo tende a ser menor) |
| failures | 0.33% | **0.06%** |
| detection_score | +1430 | **+2496** (Δ +1067, **portável** — independe de HW) |
| **final_score** | **1843** | **2742** |
| Err | 0 | **0** ✅ |

O ganho de detecção é **portável** (a fórmula não depende de hardware), então vale igual no Mac
Mini da avaliação (onde o uint8 antigo dava 1460). p99 varia por máquina.

---

## 8. Decisões de processo (como, não só o quê)

- **Brainstorm antes de codar.** Antes de qualquer linha, exploramos intenção/alternativas e só
  então spec → plano → execução. Evitou construir a coisa errada (quase fomos direto pro GBDT,
  quando a detecção tinha fix barato).
- **Medir antes de complicar.** O `cmd/diag` provou a causa (quantização da query) e o sweep provou
  que tunar IVF era beco. Cada passo: prova o gargalo → mexe → prova que melhorou.
- **Spec e plano escritos** (`docs/superpowers/specs/` e `/plans/`) com código completo e portões de
  medição — pra dar pra executar task a task e revisar.
- **Branch isolada, sem commit** (`feat/quantize-lossless-f64`) — a imagem publicada (1843/1460) não
  é tocada até a gente medir e escolher publicar.
- **Stack docker isolada** (`run-isolated.ps1`) — porque o `docker-compose.yml` de dev tinha sido
  sobrescrito com a variante de submissão (sem `build:`), e `run-test.ps1 --build` ia silenciosamente
  **puxar a imagem uint8 antiga** e medir o código errado. Build local + rede própria + caps
  espelhados, sem tocar nos containers do usuário. (Pendência: consertar o dev compose pra buildar.)
- **Portões:** cada caminho medido offline (`cmd/diag`) antes do docker, e no docker antes de
  promover; só sobe na submissão o `SCORER` com maior `final_score` (baseline a bater: 1843).

### Lições gerais transferíveis
- Conheça a função-objetivo (aqui, o log): otimize ordens de grandeza, não migalhas.
- Separe os eixos do problema (p99 vs detecção) — alavancas diferentes.
- Ataque a **causa medida**, não a suposta.
- Em precisão numérica, pergunte **qual lado** perde precisão.
- Hot path: float ok, **divisão não**; transformação fixa → tabela.
- `detection_score` é portável; `p99` não — compare só dentro do mesmo ambiente.

---

## 9. Caminho C — GBDT (em andamento)

**Por quê:** depois do Caminho A, o gargalo que sobrou é **p99** (p99_score ~247 vs teto 3000). A
detecção já está quase resolvida. Só **parar de varrer** leva o p99 a sub-ms → um **classificador
treinado** (xgboost offline → inferência Go pura) é a alavanca.

**Como (planejado):**
- treino em Python (xgboost) sobre os refs **já vetorizados** (o `.gz` guarda o vetor pronto → o
  Python não replica o `vectorize.go`); calibra um threshold τ num split de validação (sem tocar no
  `test-data.json` = sem vazamento); exporta `resources/model.json` (árvores).
- inferência `internal/model` em Go puro: anda nas árvores + sigmoid; embute o `model.json`.
- entra como `SCORER` paralelo (mantém o KNN do Caminho A como fallback forte); mede os dois.

### O que foi feito
- `train/train.py`: lê o `.gz` (vetores prontos), treina xgboost (100 árvores, depth 8,
  binary:logistic), exporta `resources/model.json` (árvores + base_score + τ) + uma amostra de
  paridade. 3M linhas, treino ~6s.
- `internal/model/model.go`: parseia as árvores e infere em Go puro (anda nas árvores + sigmoid),
  achatadas por nodeid pra walk sem ponteiro. Embutido via `go:embed`. Selecionável por `SCORER=model`.
- **Paridade Go↔xgboost: max 5.9e-8.** Pegadinha resolvida: xgboost compara em **float32**
  internamente; comparar a query em float64 fazia ~4/256 amostras de fronteira rotear pra galho
  errado (diff até 0.31). Comparar em **float32** (cast query + threshold float32) reproduziu o
  xgboost bit a bit. (Lição: pra reproduzir um modelo, **iguale a precisão da comparação dele**.)

### Resultados medidos — e a surpresa honesta

`cmd/diag` (detecção, τ-sweep): melhor **failures ~1001 (1.85%)**, `det_score` **~+369** — contra os
**34 / +2483** do IVF do Caminho A. O erro de rótulo individual (1.74%, já no round 0) é o **piso**:
o modelo treinado em rótulos individuais não reproduz bem a **maioria 5-NN** na fronteira.

Docker (`SCORER=model`, stack isolada, mesmo host contido): **p99 214ms** (vs 567ms do Caminho A),
failure_rate 1.85%, **p99_score 670**, **det_score 369**, Err=0, **final_score 1038.7**.

Duas leituras:
1. **A detecção é o que afunda** (1.85% vs 0.06% do Caminho A). det 369 vs 2496.
2. **p99 NÃO ficou sub-ms (214ms), mas CAIU de 567→214** — o compute realmente despencou (inferência
   ~800 comparações, ~µs; ~0.04 core/req, longe do cap). Os ~200ms são **piso de contenção do host**
   (10 containers competindo + cap 0.45), não o modelo. Em host limpo/dedicado (o Mac Mini do eval)
   o p99 do modelo tende a sub-ms — mas **não dá pra PROVAR isso nesta máquina contida**.

**Veredito honesto:** neste host, **Caminho A vence de longe (2742 vs 1039)**. O GBDT como está
**perde** — detecção ruim + p99 mascarado pela contenção. Foi um **sucesso de aprendizado**
(GBDT em Go puro, paridade provada, p99 caiu pela metade) e um **fracasso de pontuação**.

### Tentativa de salvar o Caminho C — e o muro confirmado (medido)

A hipótese pra salvar a detecção: **treinar por imitação do 5-NN** em vez do rótulo individual. Foi
feito: `cmd/genlabels` rotulou 1.5M pontos pelo 5-NN sobre a outra metade (IVF, sem self-match),
`train/train.py` ganhou modo `LABELS_BIN` (alvo = a *decisão 5-NN*).

| variante do modelo | failures (test-data) | det_score | obs |
|---|---|---|---|
| rótulo individual, depth 8, 100 árvores | ~1001 (1.85%) | +369 | original |
| **imitação 5-NN, depth 8, 100 árvores** | ~996 (1.84%) | **+410** | mal mexeu |
| imitação 5-NN, depth 12, 300 árvores | ~1057 (1.95%) | +408 | **overfit, pior**; json 23.6MB |

**Veredito definitivo (medido, não suposto):** o GBDT **não reproduz a fronteira 5-NN abaixo de
~1.8%** nessas 14 features — **nem** mudando o alvo (individual→imitação) **nem** a capacidade
(depth 8→12, 100→300 árvores; subir capacidade só faz overfit). A superfície de decisão do 5-NN
sobre 3M pontos é fina demais pra um ensemble de árvores axis-aligned. det do GBDT teto **~+410** vs
os **+2496** do KNN do Caminho A. A hipótese "imitação → detecção ~0.06%" foi **refutada pelo dado**.
O val-error já era plano (~1.6-1.7%) desde o round 0 — sinal de que não era underfitting nem alvo
errado, e sim **capacidade vs complexidade da fronteira**.

**Quem zera no topo** provavelmente faz **5-NN exato rápido** (o nosso brute já dá 0 failures — falta
deixá-lo sub-ms sob o cap de 0.45 CPU), não um GBDT simples. Isso é outro projeto (ANN de recall
~perfeito, SIMD no cálculo de distância, layout de cache), não uma troca de classificador.

### Conclusão da sessão
- **Caminho A é a submissão forte e segura** (1843 → 2742 local; det portável +1067; Mac Mini ~2700+
  vs 1460 atual). Robusto, sem deps novas, baixo risco.
- **Caminho C** entregou o aprendizado de ML na prática (treino xgboost, export, inferência Go pura,
  paridade bit-exata, **e duas hipóteses testadas e refutadas com dado**: imitação do 5-NN e mais
  capacidade). Lição-chave refinada: o gargalo do modelo **não** era o alvo do treino — era a
  **capacidade do ensemble de árvores vs a complexidade da fronteira 5-NN** (teto ~1.8% / det ~+410).
  Testar uma hipótese e vê-la falhar nos números é tão valioso quanto acertar. Pro topo, o caminho é
  **5-NN exato rápido** (sub-ms), não trocar de classificador.
