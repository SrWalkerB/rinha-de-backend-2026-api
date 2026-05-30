# 03 - Busca rápida (onde os timeouts morrem)

> **Pré-requisito:** [02 - Ganhos baratos](02-ganhos-baratos.md) · **Estimativa:** ~2h · **Dificuldade:** ★★★
> Troca a varredura O(3M) por um índice espacial. Ensina VP-Tree (exato) e IVF (aproximado).

## Conceito

O passo 02 provou: o custo é **algorítmico** — visitar 3.000.000 de vetores por request. A
única saída é **visitar menos**. Duas famílias de índice fazem isso:

- **VP-Tree (Vantage-Point Tree)** — árvore métrica **exata**. Usa a desigualdade triangular
  pra podar subárvores que não podem conter os 5 vizinhos. Mesmos vizinhos do brute-force →
  detecção idêntica.
- **IVF (Inverted File)** — particiona os pontos em `nlist` células (k-means). Na query, só
  varre as `nprobe` células mais próximas. **Aproximado** (o vizinho pode estar numa célula
  não visitada) → troca um pouco de precisão por muita velocidade.

Sobre **HNSW** (o que bancos vetoriais usam): foi **descartado** — o grafo (3M × ~16 vizinhos
× 4 bytes ≈ 192MB) estoura o orçamento de 150MiB por instância.

## Por que importa aqui

É o passo que tira o `final_score` do chão. Tudo antes foi preparação (medir, perfilar,
descartar atalhos). Aqui o p99 cai de ~2000ms (tudo timeout) pra requests que respondem, a
`failure_rate` despenca abaixo dos 15% do corte, e a nota vira positiva.

## Passo a passo

### 1. Manter a API congelada (drop-in)

O contrato `Score([14]float64) float64`, `NewIndex`, `Add`, `Len` **não muda** (os testes
travam isso). A estratégia entra como campos novos não-exportados em `internal/knn/knn.go`:

```go
type Index struct {
    data []uint8; bits []uint64; n int   // (inalterado)
    mode searchMode                       // brute | vptree | ivf
    vp   *vpTree
    ivf  *ivfIndex
}
```

`Score` despacha por `mode`; `Build(cfg)` constrói o acelerador **depois** de carregar os
dados. Helpers compartilhados (`dist2`, o coletor `topK`) garantem que os 3 modos produzam o
mesmo score dado o mesmo conjunto de vizinhos visitados.

### 2. Fallback para N pequeno (mantém os testes exatos)

Os testes unitários usam 5..100 vetores. Se eles caíssem no VP/IVF, a aproximação poderia
mudar resultados. Solução: `Build` só constrói o índice se `n >= buildThreshold` (2048);
abaixo disso, `mode` fica `brute`. Assim os testes rodam exatos byte-a-byte, e só os 3M reais
usam VP/IVF. Há um teste garantindo isso (`TestSmallNStaysBruteEvenWithMode`).

### 3. VP-Tree (`internal/knn/vptree.go`)

- **Build:** recursivo. Escolhe um vantage point, mede a distância dele aos outros, parte na
  mediana (raio): "inner" (mais perto) e "outer" (mais longe). Nós em array com índices
  `int32` (não ponteiros → menos GC, menos memória). Build `O(n log n)`.
- **Search:** mantém o top-K; em cada nó, decide se vale descer cada lado pela desigualdade
  triangular. Detalhe sutil: a poda usa **distância real** (com `sqrt`), porque a triangular
  não vale pra distância ao quadrado. Os `sqrt` rodam só nos nós visitados (poucos).

### 4. IVF (`internal/knn/ivf.go`)

- **Build:** k-means treinado numa **amostra** (50k pontos, limita o custo), depois atribui
  **todos** os 3M a uma célula → listas invertidas (`[][]int32`).
- **Search:** distância da query a cada centroide → pega os `nprobe` mais próximos → varre só
  essas listas. Os centroides e o buffer de `nprobe` ficam na pilha (zero alocação por query).
- **Knobs:** `nlist` (nº de células), `nprobe` (células varridas), via env.

### 5. Seletor por env + Build no startup

`main.go`, na goroutine de load, após `dataset.Load`:
```go
cfg := knn.BuildConfig{
    Mode:   getenv("KNN_INDEX", "ivf"),
    NList:  atoiEnv("KNN_NLIST", 256),
    NProbe: atoiEnv("KNN_NPROBE", 8),
    Iters:  atoiEnv("KNN_KMEANS_ITERS", 8),
}
ix.Build(cfg)   // <- só aqui; antes de s.index.Store
```
`KNN_INDEX=vptree|ivf|brute` permite comparar os três sem recompilar.

### 6. Regressão
```powershell
go test ./...   # verde: VP exato == brute, IVF recall >= 0.90, small-N fica brute
```

## Como medir

Micro (rápido):
```powershell
go test -bench=Score -benchmem -run=^$ ./internal/knn
```
Sistema (real):
```powershell
.\run-test.ps1 -SkipSmoke
```

## Resultado medido

### Benchmark (200k vetores, na máquina local, sem cap)

| modo | ns/op | vs brute |
|------|-------|----------|
| Brute | 1.61 ms | 1× |
| **VP-Tree** | **2.32 ms** | **0.7× — MAIS LENTO!** |
| **IVF** (nlist 256, nprobe 8) | **0.108 ms** | **~15× mais rápido** |

**A surpresa do VP-Tree:** ficou *mais lento* que o brute-force. Em **14 dimensões** a poda da
desigualdade triangular é fraca (maldição da dimensionalidade): a árvore visita quase todos os
nós e ainda paga `sqrt` + recursão + saltos de memória. Lição: método exato sofisticado **não
é garantia** de ganho em dimensão moderada. **IVF (aproximado) é o vencedor.**

### Sistema — IVF no dataset real de 3M (docker, sob os limites)

A jornada de tuning (cada linha = uma rodada do `run-test.ps1`):

| config | p99 | failure_rate | Err | final_score |
|--------|-----|--------------|-----|-------------|
| baseline brute-force | 2002ms | 100% | 13830 | **-6000** |
| ivf nlist=1024 nprobe=12, cpu 0.40 | 2002ms | 7.86% | 1583 | **-3751** |
| ivf nprobe=8, cpu 0.45 | 1515ms | 0.63% | 45 | **+778** |
| ivf nprobe=6, GOGC off (passo 04) | 1202ms | 0.51% | **0** | **+1136** |

A primeira rodada IVF já tirou a detecção do corte (100%→7.86% de falha) — IVF preserva a
detecção (FP/FN minúsculos). As rodadas seguintes (passo 04) deram folga de throughput até
**zerar os timeouts** (`Err: 0`, 100% das requests atendidas).

## Armadilhas

- **Não testar a corretude do índice** — VP/IVF têm bug fácil. O teste compara contra o
  brute-force (fonte da verdade): VP tem que bater **exato**; IVF, recall alto.
- **Empates de distância** — comparar `fraud_score` entre métodos falha em empates (ordem de
  inserção difere). Compare o **multiset de distâncias** dos K vizinhos.
- **VP-Tree com ponteiros** — nós com ponteiros explodem GC/memória em 3M. Use arrays + `int32`.
- **`nlist` grande demais** — build `O(n·nlist)` no startup, sob 0.40 CPU, fica lento. Ver
  passo 05 (construir no build).
- **nprobe alto demais** — mais lento sem ganho real de detecção (FP/FN já são minúsculos).

## Registro

Já lançado na tabela do [README.md](README.md). Marco: `final_score` saiu do piso (-6000) e
**virou positivo** (+1136), com 0 erros HTTP.

## Próximo

[04 - Tuning de runtime](04-tuning-runtime.md): os knobs (`GOGC`, CPU) que levaram de
+778 a +1136 — e por que o p99 ainda não é single-digit (entra o passo 05).
