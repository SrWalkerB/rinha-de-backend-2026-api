# 02 - Ganhos baratos (e por que não bastam)

> **Pré-requisito:** [01 - Perfilar o gargalo](01-perfil-do-gargalo.md) · **Estimativa:** ~30 min · **Dificuldade:** ★★☆
> Ensina **benchmark do Go** e a disciplina de medir micro-otimizações — incluindo aceitar quando não pagam.

## Conceito

"Ganho barato" = mudança pequena e local que talvez acelere sem reescrever o algoritmo:
parar uma conta mais cedo, reaproveitar memória, trocar uma função por uma mais leve. São
tentadoras porque parecem grátis. Mas a regra do doc 01 vale aqui também: **só conta se o
benchmark provar.** Neste passo você aprende a medir no nível micro (benchmark do Go) e vê,
na prática, que a otimização "óbvia" pode não ajudar — ou até atrapalhar.

## Por que importa aqui

O profile do passo 01 já nos disse muita coisa:
- `knn.(*Index).Score` = **98% do CPU**, no loop de distância (`internal/knn/knn.go` ~98-100).
- `encoding/json` e alocações = **ruído** (nem apareceram).

Isso **descarta** dois "ganhos baratos" clássicos antes de escrevê-los: otimizar JSON ou
reduzir alocações **não move nada** quando eles custam ~0% do tempo. Esse é o valor de medir:
o profile nos salvou de horas otimizando o que não importa. Sobra um candidato que toca o
hotspot real: **early-abandon** no loop de distância.

## Passo a passo

### 1. Aprender a medir micro: benchmark do Go

O teste de carga (k6) mede o sistema inteiro e leva ~2 min por rodada. Pra iterar numa função
específica existe coisa mais rápida: o **benchmark** embutido do Go. Criamos
`internal/knn/knn_bench_test.go`:

```go
func BenchmarkScore(b *testing.B) {
    ix := buildBenchIndex(200_000)   // índice sintético determinístico (seed fixa)
    // ... query fixa ...
    b.ReportAllocs()
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = ix.Score(q)              // mede só o Score, repetido b.N vezes
    }
}
```

Conceitos:
- `b.N` — o Go roda o loop quantas vezes precisar pra ter uma medida estável; você não escolhe.
- `b.ResetTimer()` — zera o cronômetro DEPOIS do setup (construir o índice não conta).
- `b.ReportAllocs()` — reporta `B/op` e `allocs/op` (memória por operação).
- seed fixa (`rand.NewSource(42)`) — mesmos dados todo run → números comparáveis.

Rodar:
```powershell
$env:Path += ";C:\Program Files\Go\bin"
cd C:\Me\me-projects\rinha-backend-2026\rinha-de-backend-2026-api
go test -bench=Score -benchmem -run=^$ -count=3 ./internal/knn
```
(`-run=^$` desliga os testes normais; `-count=3` roda 3× pra ver a variância.)

**Baseline medido:** `~1.7 ms/op`, **`0 allocs/op`** (200k vetores). O `0 allocs` confirma o
profile: `Score` não aloca nada. Pra 3M escala ~15× → ~25ms. Bate com a estimativa.

### 2. Hipótese: early-abandon

A distância é uma soma de quadrados que só **cresce**. Se a soma parcial já passou da
distância do 5º vizinho mais próximo atual (`bestDist[K-1]`), aquele ponto **não** entra no
top-5 — dá pra parar de somar as dimensões restantes. Em teoria, menos contas.

A mudança (no loop interno de `Score`): adicionar `if dist >= worst { break }` dentro do
`for d`.

### 3. Medir — e a surpresa

```
baseline (sem early-abandon):  ~1.60-1.87 ms/op
com early-abandon:             ~1.72-2.00 ms/op   ← IGUAL ou PIOR
```

**Não ajudou. Piorou de leve.** Por quê (a lição):
- O loop interno tem só **14 iterações** — curtíssimo. O `if` extra a cada iteração custa
  quase tanto quanto a conta que ele evita.
- O branch **quebra o pipelining/vetorização** que a CPU fazia no loop apertado.
- Com **14 dimensões e dados espalhados** (maldição da dimensionalidade), a soma parcial
  raramente estoura o limite nas primeiras dimensões → o `break` quase nunca dispara, mas o
  custo do branch é pago em **todas** as 3M iterações.

### 4. Decisão: reverter

Benchmark é o juiz. Não pagou → **revertemos** (o `Score` voltou ao original). `go test ./...`
verde (a mudança era exata, mas a gente nem manteve).

> Nuance honesta: o benchmark usa dados uniformes sintéticos. No dataset real (3M, mais denso,
> com a estrutura do sentinela -1) o early-abandon **poderia** disparar mais. Mas o ganho
> máximo seria pequeno e não muda o veredito: o problema é **O(N)**, não a constante. Gastar
> mais tempo aqui é otimizar a coisa errada.

### 5. Regressão
```powershell
go test ./...   # verde — nada de lógica mudou
```

## Como medir

Micro: `go test -bench=Score`. Sistema: `run-test.ps1`. Neste passo **nenhuma mudança foi
mantida**, então não há linha nova na tabela de progresso — o `final_score` no docker segue
-6000. Isso é o ponto.

## Resultado esperado

- Você sabe escrever e rodar um benchmark do Go e ler `ns/op` / `allocs/op`.
- Você viu, com número, que: (a) JSON/allocs são ruído aqui (profile), (b) early-abandon não
  paga neste formato. 
- **A conclusão que destrava o passo 03:** nenhum ganho barato resolve, porque o custo é
  **algorítmico** — visitar 3.000.000 de vetores por request. A única saída é **visitar menos
  vetores**. É exatamente isso o próximo passo.

## Armadilhas

- **Otimizar sem medir** — teria "consertado" o JSON e comemorado 0% de ganho.
- **Acreditar na intuição contra o benchmark** — early-abandon "deveria" ajudar; não ajudou.
- **Benchmark não-determinístico** — sem seed fixa, cada run muda e nada é comparável.
- **Cronometrar o setup** — esquecer `b.ResetTimer()` mede a construção do índice junto.
- **Generalizar de dados sintéticos** — anote a ressalva; confirme no real quando for decisivo.

## Registro

Sem linha de progresso (nenhuma mudança mantida). O aprendizado fica neste doc; o benchmark
(`knn_bench_test.go`) **permanece** — vai medir o ganho real do passo 03.

## Próximo

[03 - Busca rápida](03-busca-rapida.md): trocar a varredura O(3M) por um índice espacial
(VP-Tree exato + IVF aproximado). **Aqui os timeouts morrem.**
