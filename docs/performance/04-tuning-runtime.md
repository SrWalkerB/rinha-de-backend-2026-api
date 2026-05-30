# 04 - Tuning de runtime

> **Pré-requisito:** [03 - Busca rápida](03-busca-rapida.md) · **Estimativa:** ~45 min · **Dificuldade:** ★★☆
> Os knobs do Go e do docker-compose que levaram a nota de +778 a +1136.

## Conceito

Depois que o algoritmo está certo (passo 03), sobram os **botões** do ambiente: quanto CPU
cada container recebe, como o Go gerencia memória/GC, quantas threads usa. São ganhos de
"último quilômetro" — pequenos sozinhos, somam quando você está no limite da capacidade.

## Por que importa aqui

Depois do IVF, a API ficou **bem no limite** de 900 req/s: a primeira config IVF dava p99
2002ms (ainda no corte) e 7.86% de falha. Não era mais o algoritmo — era **falta de folga**.
Estes knobs deram essa folga, zerando os timeouts.

## Os knobs

### `GOMAXPROCS=1`
Número de threads do Go que rodam goroutines em paralelo. Com `cpus: 0.45` (menos de 1 core),
mais de 1 thread só gera troca de contexto e contenção. **1 é o certo** aqui. (Já estava
setado.)

### `GOMEMLIMIT=150MiB`
Limite **suave** de memória do Go. O GC se esforça pra manter o heap abaixo disso. Casado com
o limite **rígido** do container (155MB) — se o Go ignorasse, o container levaria OOM-kill.
Nosso live set (vetores 42MB + listas IVF 12MB ≈ 54MB) cabe folgado.

### `GOGC=off` ⭐ (o ganho desta etapa)
`GOGC` controla **com que frequência** o GC roda: default 100 = coletar quando o heap dobra.
Com muitas alocações curtas (o JSON de cada request), isso dispara GC toda hora → CPU gasta em
GC em vez de servir.

`GOGC=off` desliga esse gatilho — o GC passa a disparar **só** quando chega perto do
`GOMEMLIMIT`. Como o live set (54MB) está bem abaixo de 150MiB, o GC roda **pouquíssimo**
durante o teste. Quase toda a CPU vai pra servir requests. É o idioma "soft memory limit":
`GOGC=off` + `GOMEMLIMIT` = throughput máximo dentro de um teto de memória.

> Seguro porque o live set é pequeno e estável. Se ele crescesse perto de 150MiB, `GOGC=off`
> causaria GC em loop (thrash). Sempre cheque a folga antes.

### Rebalance de CPU (docker-compose)
Soma de CPU é fixa (≤ 1.0 no total). A detecção/busca roda nas **APIs**, não no nginx (que só
faz proxy de payloads pequenos). Então tiramos do nginx e demos pras APIs:

```yaml
nginx: cpus "0.20" -> "0.10"
api1/api2: cpus "0.40" -> "0.45"   # 0.45 + 0.45 + 0.10 = 1.00
```
+12.5% de CPU por instância = +12.5% de capacidade, de graça (o nginx nem sentiu).

## Como medir

```powershell
.\run-test.ps1 -SkipSmoke
```
Mude **um** knob por vez pra atribuir o ganho. (Aqui, por velocidade, GOGC e nprobe mudaram
juntos; o ideal didático é isolar.)

## Resultado medido

| mudança | p99 | failure_rate | Err | final_score |
|---------|-----|--------------|-----|-------------|
| IVF nprobe=12, cpu 0.40, GOGC default | 2002ms | 7.86% | 1583 | -3751 |
| + nprobe=8, cpu 0.45/0.10 | 1515ms | 0.63% | 45 | +778 |
| + nprobe=6, **GOGC=off** | 1202ms | 0.51% | **0** | **+1136** |

Os timeouts **zeraram** (`Err: 0`) — 100% das requests atendidas.

## Por que o p99 ainda é 1202ms (e não 10ms)

A nota de latência ainda é levemente negativa (`p99_score ≈ -80`) porque o p99 fica em ~1.2s.
O motivo: o sistema roda **bem no limite** de 900 req/s. Quando a carga chega no pico, forma-se
uma fila curta, e o percentil 99 captura essa cauda. Não é timeout (tudo responde < 2001ms),
mas é alto.

Pra derrubar o p99 pra single/double-digit ms (e ganhar +1000~+2000 de `p99_score`), precisa
de **muito mais folga de capacidade** — o que pede um `nlist` maior (células menores → query
mais rápida). Mas construir um `nlist` grande no startup, sob 0.45 CPU, é lento (já leva ~85s).
A saída é construir o índice **fora** do runtime → próximo passo.

## Armadilhas

- **GOGC=off com live set grande** — vira GC em loop. Só use com folga de memória conferida.
- **Mais threads que CPU** — `GOMAXPROCS` alto com `cpus<1` só piora (contenção).
- **Mudar vários knobs juntos** — não dá pra saber qual ajudou. Isole quando possível.
- **Esquecer que a soma de CPU é limitada** — dar mais a um serviço tira de outro.

## Registro

Lançado na tabela do [README.md](README.md). Marco: **0 timeouts**, `final_score +1136`.

## Próximo

[05 - Pré-processamento no build](05-preprocessamento-no-build.md): construir o índice no
Docker build (CPU cheia, fora do cap) → startup quase instantâneo **e** liberdade pra usar
`nlist` grande → p99 despenca.
