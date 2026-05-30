# NN - Título da técnica

> **Pré-requisito:** doc anterior · **Estimativa:** ~X min · **Dificuldade:** ★☆☆

## Conceito

O que é a técnica, em linguagem simples. Uma analogia se ajudar. (3-6 linhas.)

## Por que importa aqui

Como ESTE gargalo se beneficia. Citar o arquivo/função exato (ex.: `internal/knn/knn.go`
`Score()`) e o número atual da tabela de progresso. (3-6 linhas.)

## Passo a passo

A menor mudança possível, uma de cada vez. Qual arquivo/função tocar e o que fazer
(descrever — você escreve o código). Sempre terminar com a regressão:

```powershell
$env:Path += ";C:\Program Files\Go\bin"
go test ./...    # tem que continuar verde
```

## Como medir

Comandos exatos (PowerShell, Go em `C:\Program Files\Go\bin`). Como gerar carga e onde ler
o número. Normalmente:

```powershell
.\run-test.ps1
```

## Resultado esperado

O delta esperado: `p99 X→Y`, `final_score A→B`, ou "flame graph mostra `Score()` encolhendo".
Se não bater, é sinal de que a hipótese estava errada — investigar antes de seguir.

## Armadilhas

- Comparar runs de ambientes diferentes (cap vs sem cap).
- Não esperar as duas instâncias ficarem `ready`.
- Olhar `avg` em vez de `p99`.
- Quebrar um teste travado.

## Registro

Linha(s) a adicionar na tabela de progresso do [README.md](README.md):

```
| NN | <técnica> | <data> | <p99> | <failure_rate> | <p99_score> | <detection_score> | <final_score> | <Δ> | <ambiente> | <notas> |
```
