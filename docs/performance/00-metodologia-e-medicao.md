# 00 - Metodologia e medição

> **Pré-requisito:** nenhum · **Estimativa:** ~30 min de leitura + 1 run · **Dificuldade:** ★☆☆
> **Não muda código.** É a fundação: aprender a *medir* antes de otimizar.

## Conceito

Otimizar sem medir é chutar no escuro. Três perguntas guiam todo o resto da trilha:

1. **Quanto está rápido/lento?** → teste de carga (k6) te dá o `p99` e a vazão.
2. **Quão boa é a nota?** → a fórmula de score traduz latência + acerto em pontos.
3. **Onde exatamente está o gargalo?** → profiling (pprof) aponta a linha que custa CPU.

Este doc cobre as três ferramentas no nível conceitual. Os próximos docs *aplicam*.

---

## Parte 1 — Teste de carga com k6

"Funciona no Postman" não quer dizer "aguenta produção". Uma request isolada da nossa API
responde em ~25ms — tranquilo. Mas o avaliador manda **900 requests por segundo** por 2
minutos. É outro mundo. O teste de carga simula isso.

A ferramenta é o **k6**. O script oficial fica em `../../../rinha-de-backend-2026/test/test.js`.
Os conceitos-chave (todos visíveis no `options` do script):

### Vocabulário

- **VU (Virtual User)** — um "usuário virtual", uma thread que dispara requests em loop.
  O teste usa até `maxVUs: 250`.
- **Iteração** — uma execução da função de teste = **um `POST /fraud-score`**. O script lê
  uma transação rotulada do `test-data.json` e compara a resposta da API com o gabarito.
- **Arrival-rate (modelo aberto)** — o detalhe mais importante. O executor é
  `ramping-arrival-rate`: a carga chega num **ritmo fixo** (`stages: target 900` req/s),
  **independente** de a API dar conta ou não. Se a API trava, as requests **não esperam na
  fila do cliente** — elas continuam chegando e **estouram**. É por isso que um servidor
  lento não fica "só mais devagar": ele entra em colapso (100% timeout). Modelo aberto é
  cruel de propósito — é como o mundo real funciona.
- **timeout `2001ms`** — cada request espera no máximo 2001ms. Passou disso, vira erro.
  Esse número não é aleatório: bate com o corte de p99 da fórmula (`p99 > 2000ms` zera a nota
  de latência).
- **stages** — `{ duration: '120s', target: 900 }` = sobe de 1 a 900 req/s ao longo de 120s.
  Rampa, não tudo de uma vez — assim dá pra ver em que ponto a API começa a falhar.

### p99 vs avg vs p50 — por que a CAUDA importa

- **avg (média)** — engana: 99 requests de 5ms + 1 de 2000ms dá média ~25ms, parecendo OK.
- **p50 (mediana)** — metade é mais rápida, metade mais lenta. Ignora os piores casos.
- **p99 (percentil 99)** — 99% das requests terminam *abaixo* desse tempo; só o 1% pior fica
  acima. **É o que a Rinha avalia.** Mede a experiência ruim, não a média feliz. Um p99 alto
  significa que muita gente sentiu lentidão — mesmo que a média esteja boa.

O script pede exatamente isso: `summaryTrendStats: ['p(99)']`.

### thresholds vs scoring (duas coisas diferentes)

- **thresholds** (no `smoke.js`) — portões de pass/fail. Ex.: `checks: ['rate==1.0']` =
  "100% dos checks têm que passar". Servem pra validar rápido que a API responde certo.
- **scoring** (no `test.js`, função `handleSummary`) — a NOTA de verdade, calculada no fim do
  teste de carga. É o que a Parte 2 explica.

---

## Parte 2 — A fórmula de score

A nota final vai de **-6000 a +6000** e é a soma de duas metades independentes:

```
final_score = p99_score + detection_score
```

(Constantes reais, lidas de `k6-summary.js`: `K=1000`, `T_MAX=1000ms`, `P99_MIN=1ms`,
`P99_MAX=2000ms`, `EPSILON_MIN=0.001`, `BETA=300`, corte de falha = 15%.)

### Metade 1 — latência (`p99_score`), de -3000 a +3000

```
se p99 > 2000ms:  p99_score = -3000          ← CORTE (latência inviável)
senão:            p99_score = 1000 · log10(1000 / max(p99, 1))
```

É **logarítmico** — cada 10× mais rápido vale +1000 pontos fixos:

| p99 | p99_score |
|-----|-----------|
| ≤ 1 ms | **+3000** (teto, satura) |
| 10 ms | +2000 |
| 100 ms | +1000 |
| 1000 ms | 0 |
| > 2000 ms | **-3000** (corte) |

Lição: vale muito caçar latência até ~1ms; abaixo disso não rende mais.

### Metade 2 — detecção (`detection_score`), de -3000 a +3000

Primeiro, classifica cada resposta numa de 5 caixas (o `test.js` faz isso, linhas 65-79):

| caixa | significado |
|-------|-------------|
| **TP** | fraude corretamente negada (`approved:false` numa fraude) |
| **TN** | legítima corretamente aprovada (`approved:true` numa legítima) |
| **FP** | legítima negada por engano (bloqueou cliente bom) |
| **FN** | fraude aprovada por engano (deixou passar) |
| **Err** | resposta != HTTP 200 (timeout, 500…) |

Depois:
```
E            = 1·FP + 3·FN + 5·Err          (erros ponderados — Err é o pior, peso 5)
ε            = E / N
failure_rate = (FP + FN + Err) / N

se failure_rate > 15%:  detection_score = -3000      ← CORTE (rígido)
senão:                  detection_score = 1000·log10(1/max(ε,0.001)) − 300·log10(1+E)
```

Lições:
- **Err pesa 5** (FN pesa 3, FP pesa 1). Devolver erro HTTP é o pior pecado — pior que errar
  a detecção. Por isso a API tem fallback que sempre responde 200.
- **Corte de 15% é rígido.** Passou de 15% de falha (FP+FN+Err), a metade de detecção vira
  -3000 e nenhum p99 bom compensa. Sair da zona de corte é prioridade #1.

### Lendo o `results.json`

Cada campo mapeia num termo:
```json
{
  "p99": "2002.04ms",                       // entra em p99_score
  "scoring": {
    "breakdown": { ...TP/TN/FP/FN/Err... }, // as 5 caixas
    "failure_rate": "100%",                 // (FP+FN+Err)/N → testa o corte de 15%
    "weighted_errors_E": 69150,             // E = 1·FP+3·FN+5·Err
    "error_rate_epsilon": 5,                // ε = E/N
    "p99_score":       { "value": -3000, "cut_triggered": true },
    "detection_score": { "value": -3000, "cut_triggered": true },
    "final_score": -6000
  }
}
```

---

## Parte 3 — Profiling com pprof (intro)

O teste de carga diz *que* está lento. O **profiler** diz *onde*. O Go tem um embutido: o
**pprof**. Dois tipos que vamos usar:

- **CPU profile** — amostra, várias vezes por segundo, qual função está rodando. Resultado:
  "X% do tempo de CPU foi gasto na função Y". É a estrela pra um problema de CPU como o nosso
  (varrer 3M vetores). *Onde o processador sua.*
- **Heap profile** — onde a memória é alocada. Útil no passo 02 (reduzir alocações por request).

### Como ler um profile

- **`top`** — ranking das funções por custo. A #1 é o gargalo.
- **`top -cum`** (cumulative) — inclui o tempo das funções chamadas por baixo. Útil pra ver
  "esse caminho inteiro custa X".
- **self vs cumulative** — *self* = tempo gasto NA própria função; *cumulative* = ela + tudo
  que ela chama. Um loop apertado tem *self* alto.
- **`list <função>`** — mostra o código-fonte da função com o custo **linha a linha**. É onde
  você vê qual linha exata dói.
- **`web` / flame graph** — desenho: cada caixa é uma função, a **largura = custo**. A caixa
  mais larga é o gargalo. (Precisa do Graphviz `dot`; senão exporta `-svg`/`-png`.)

No passo 01 a gente liga o pprof na API e captura isso ao vivo. A previsão: `knn.(*Index).Score`
vai dominar ~80-90% do CPU — provando, com dado e não com achismo, que é ali que mora o
problema.

---

## Mão na massa (agora)

Roda o baseline tu mesmo e olha cada número à luz da fórmula:

```powershell
cd C:\Me\me-projects\rinha-backend-2026\rinha-de-backend-2026-api
.\run-test.ps1
```

Quando terminar, abre `..\rinha-de-backend-2026\test\test\results.json` e responde:
1. Qual dos dois cortes disparou? (dica: os dois.)
2. Quanto do `final_score` é latência e quanto é detecção?
3. Por que `TP/TN/FP/FN` estão todos em 0? (dica: nenhuma request chegou a responder 200.)

## Resultado esperado

Você consegue rodar o teste de carga, ler o `results.json` e explicar, com a fórmula na mão,
por que a nota é -6000. Nenhuma linha de código mudou ainda. A tabela de progresso no
[README.md](README.md) já tem a linha `baseline`.

## Armadilhas

- Rodar o k6 no diretório errado (ver `TESTING.md`) — usa o `run-test.ps1`, ele cuida disso.
- Não esperar as **duas** instâncias ficarem `ready` antes de medir.
- Olhar a média (`avg`) em vez do `p99`.
- Comparar runs de ambientes diferentes (com cap de CPU vs sem) — sempre anote a coluna
  `ambiente` na tabela.

## Próximo

[01 - Perfilar o gargalo](01-perfil-do-gargalo.md): ligar o pprof e provar onde está o custo.
