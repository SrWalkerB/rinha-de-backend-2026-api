# 06 - SIMD (avançado, opcional)

> **Pré-requisito:** [05 - Pré-processamento no build](05-preprocessamento-no-build.md) · **Dificuldade:** ★★★★
> Status: **conceito / opcional.** Só vale se, após 03-05, o cálculo de distância ainda for o gargalo.
>
> ⚠️ **`GOAMD64=v3` está propositalmente DESLIGADO no `Dockerfile`.** Ele exige AVX2 na CPU —
> se a máquina de avaliação não tiver, o binário morre com `SIGILL`. Como não sabemos o host
> da Rinha, fica como experimento manual (abaixo), não no build commitado.

## Conceito

**SIMD** (Single Instruction, Multiple Data) = uma instrução da CPU opera sobre **vários
dados de uma vez**. Um registrador AVX2 tem 256 bits = 32 bytes → dá pra subtrair/multiplicar
**16 ou 32 dimensões uint8/int16 numa tacada**, em vez de uma por iteração do `for d`.

Nossa distância é exatamente isso: `Σ (q[d] - data[d])²` sobre 14 dimensões. Um kernel SIMD
faria as 14 subtrações + quadrados + soma em poucas instruções → potencial 4-8× no
`dist2`/`centroidDist`.

## Por que (talvez) importe aqui

Depois do passo 05 (índice pré-construído, `nlist` grande), a query visita poucos milhares de
linhas, mas a maior parte do tempo restante ainda é o loop de distância. Se o profile (passo
01, refeito) mostrar `dist2`/`centroidDist` dominando de novo, SIMD é o último ganho.

> **Meça antes.** Pode ser que, após 03-05, o gargalo já seja o JSON/HTTP, não a distância —
> aí SIMD não ajuda (volta pro espírito do passo 02). O profile decide.

## O problema em Go (honesto)

Go **não** tem intrínsecos SIMD no código normal. Opções, todas com custo:

1. **Assembly (`.s`)** — escrever o kernel em assembly Plan9 do Go (`internal/knn/dist_amd64.s`
   + stub). Funciona, mantém o binário estático (`CGO_ENABLED=0`), mas é difícil e específico
   de arquitetura (precisa de fallback Go puro pra não-amd64).
2. **`GOAMD64=v3`** no build — habilita o compilador a usar AVX2 automaticamente em alguns
   loops. Ganho modesto e não garantido, mas **trivial** (só uma env no `go build`). Bom
   primeiro experimento.
3. **Biblioteca SIMD** (ex.: geradores como `avo`) — gera o assembly por você. Adiciona
   dependência/complexidade ao build.
4. **CGO + intrínsecos C** — proibido aqui (quebra o binário estático e o espírito stdlib).

## Passo a passo (se for fazer)

1. Refazer o profile (passo 01) **depois** de 03-05. Confirmar que distância ainda domina.
2. Tentar primeiro o **barato**: `GOAMD64=v3` no `go build` do Dockerfile; medir.
3. Se valer mais, escrever um kernel assembly para `dist2` de 14×uint8, com fallback Go puro
   atrás de uma build tag. Testar contra a versão Go (resultados idênticos).
4. Medir com benchmark (`go test -bench`) e com `run-test.ps1`.

## Armadilhas

- **Otimizar sem profile** — clássico. Pode estar otimizando o que não é mais o gargalo.
- **Sem fallback de arquitetura** — assembly amd64 quebra o build em arm64.
- **Complexidade vs ganho** — SIMD é o item de pior relação esforço/retorno da trilha. Só
  encare se os passos anteriores já esgotaram e o profile aponta a distância.

## Registro

(Se implementado.) Esperado: ganho no `p99` proporcional ao quanto a distância pesava no
profile pós-05.

## Fim da trilha

Voltar ao [README.md](README.md) e olhar a tabela de progresso completa: de **-6000** (tudo
timeout) até a nota atual. Cada linha é uma técnica que você entende porque mediu o antes e o
depois.
