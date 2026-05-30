# 01 - Perfilar o gargalo

> **Pré-requisito:** [00 - Metodologia](00-metodologia-e-medicao.md) · **Estimativa:** ~30 min · **Dificuldade:** ★★☆
> Liga o pprof, captura um CPU profile sob carga e **prova** onde está o gargalo.

## Conceito

No doc 00 vimos que o `final_score` é -6000 e *suspeitamos* que o vilão é a busca
força-bruta. Suspeitar não basta — **vamos provar**. O profiler (pprof) amostra o programa
rodando e diz "X% do tempo de CPU foi gasto na função Y". Se a hipótese estiver certa,
`knn.(*Index).Score` vai aparecer comendo ~80-90% do CPU. Se não estiver, melhor descobrir
agora, antes de reescrever a coisa errada.

## Por que importa aqui

A regra de ouro de performance: **otimize o que o profile aponta, não o que você acha.**
Programadores erram o palpite o tempo todo (acham que é o JSON, é o loop; acham que é a rede,
é a CPU). Este passo te dá o hábito que vale pra qualquer API daqui pra frente: medir →
localizar → só então mexer. O alvo provável é `internal/knn/knn.go` `Score()` (linhas 77-122),
o loop que varre 3M vetores.

## Passo a passo

### 1. O pprof já está fiado no código

Em `main.go` foram adicionados (e os testes seguem verdes):

- O import em branco `_ "net/http/pprof"` — o `init()` desse pacote registra as rotas
  `/debug/pprof/*` no `http.DefaultServeMux` (o mux padrão do Go).
- Um servidor de debug **separado**, ligado **só** quando a env `PPROF_ADDR` está setada:

  ```go
  if pprofAddr := getenv("PPROF_ADDR", ""); pprofAddr != "" {
      go func() { http.ListenAndServe(pprofAddr, nil) }() // nil = DefaultServeMux (pprof)
  }
  ```

Por que porta separada e gated por env:
- **Separada** — o pprof NÃO fica na porta da API (`:8080`/`:9999`), então nunca vaza pro
  nginx nem pro k6.
- **Gated** — produção/submissão não setam `PPROF_ADDR` → o servidor de debug nem sobe. Zero
  risco na nota, zero exposição.

### 2. Suba a API local com pprof ligado

Profilamos **local** (`go run`), não no Docker. Motivo: o build de produção usa
`-ldflags="-s -w"` (tira o DWARF) e roda em distroless (sem fonte/toolchain), então `list`
linha-a-linha não funciona lá. Local tem símbolos e fonte intactos, e sem o cap de 0.40 CPU
o hotspot aparece mais rápido. (O número que vale continua sendo o do Docker via `run-test.ps1`;
aqui o objetivo é *localizar*, não cronometrar.)

Janela 1 — sobe a API:
```powershell
$env:Path += ";C:\Program Files\Go\bin"
cd C:\Me\me-projects\rinha-backend-2026\rinha-de-backend-2026-api
$env:ADDR=":8088"          # qualquer porta livre; 8080 pode estar ocupada por outro app
$env:PPROF_ADDR="127.0.0.1:6060"
go run .
```
Espera o log `ready: 3000000 reference vectors loaded` (~8s) e `pprof listening on 127.0.0.1:6060`.
(Se trocar a porta da API, ajuste o `-Url` do hammer no passo 3.)

> Se o dataset não estiver em `resources/`, aponte: `$env:REFERENCES_PATH="..\rinha-de-backend-2026\resources\references.json.gz"` antes do `go run .`.

### 3. Gere carga (pra ter o que perfilar)

O profile só faz sentido **enquanto há tráfego**. Janela 2 — martela o endpoint:
```powershell
cd C:\Me\me-projects\rinha-backend-2026\rinha-de-backend-2026-api
.\hammer.ps1 -Seconds 40
```
Confirme que **um core fica perto de 100%** (Gerenciador de Tarefas). Sem isso, o profile sai
ocioso (só runtime).

### 4. Capture e leia o CPU profile

Janela 3 — enquanto o hammer roda, capture 30s. A forma mais amigável abre uma UI web com
**flame graph**:
```powershell
$env:Path += ";C:\Program Files\Go\bin"
go tool pprof -http=:8081 "http://localhost:6060/debug/pprof/profile?seconds=30"
```
Abre o navegador em `:8081`. Veja:
- **Flame Graph** (menu View) — cada caixa é uma função, **largura = custo**. A caixa larga é
  o gargalo.
- **Top** — ranking por custo. A #1 é o vilão.
- **Source** — código da função com custo por linha.

Sem navegador/Graphviz? Modo texto (mesma captura):
```powershell
go tool pprof "http://localhost:6060/debug/pprof/profile?seconds=30"
# no prompt (pprof):
#   top           -> ranking por self time
#   top -cum      -> ranking incluindo o que cada função chama
#   list Score    -> custo linha-a-linha de knn.(*Index).Score
#   svg > docs\performance\assets\cpu-baseline.svg   (precisa do Graphviz 'dot')
```

### 5. (Opcional) heap profile

Pra ver alocações por request (relevante no passo 02):
```powershell
go tool pprof "http://localhost:6060/debug/pprof/heap"
```

### 6. Regressão

Nada de lógica mudou, mas confirme:
```powershell
$env:Path += ";C:\Program Files\Go\bin"
go test ./...
```

## Como medir

Este passo **não muda a nota** — o endpoint pprof é inerte sob carga real e a lógica é a
mesma. A "medição" aqui é o próprio profile: a evidência de qual função custa CPU. Salve o
flame graph/SVG em `docs/performance/assets/` como registro do baseline.

## Resultado esperado

- `top` / flame graph dominados por **`rinha-fraud/internal/knn.(*Index).Score`** — esperado
  ~80-90% do CPU.
- `list Score` atribui a maior parte do *self time* ao loop interno
  (`dist += uint32(diff*diff)`, knn.go ~98-101) — ou seja, à varredura dos 3M × 14.
- `encoding/json` e o GC aparecem como custo **secundário** (guarda isso pro passo 02).
- Conclusão provada: o gargalo é **algorítmico** (O(N)), não micro. Isso justifica o passo 03
  (trocar a busca). Ganhos baratos (passo 02) ajudam, mas não resolvem sozinhos.

> 📌 **Já capturado** neste projeto: ver [`assets/cpu-baseline.md`](assets/cpu-baseline.md) —
> `Score()` deu **98,48% do CPU**, com o loop de distância (linhas 98-100) somando ~84s de 88s.

## Armadilhas

- **Profile ocioso** — capturar sem o hammer rodando. Confirme 1 core ~100% antes.
- **Janela de captura não sobrepõe o tráfego** — os 30s do profile têm que cair durante os
  40s do hammer.
- **`web`/`svg` reclamando** — falta o Graphviz (`dot`). Use a UI `-http` (flame graph não
  precisa de dot) ou só `top`/`list`.
- **Perfilar o Docker esperando ver linhas** — binário stripped + distroless não dá `list`.
  Local pra localizar; Docker só pra confirmar o efeito do cap (nível função).
- **Esquecer de tirar o pprof da submissão** — checklist abaixo.

### ✅ Checklist submission-safe

- [ ] `PPROF_ADDR` **não** aparece no `docker-compose.yml` nem no `Dockerfile`.
- [ ] Nenhuma porta de debug (ex.: `6060`) publicada no compose.
- [ ] `go test ./...` verde.

(O import `_ "net/http/pprof"` pode ficar — sem `PPROF_ADDR` o servidor não sobe. Se quiser
zero resíduo, dá pra mover pra um arquivo com build tag `//go:build pprof` num passo futuro.)

## Registro

Profiling não rende linha de nota. Se quiser registrar que rodou, use `ambiente=local-pprof`
e anote o hotspot em `notas` (ex.: "Score() ~85% CPU"). O artefato real é o SVG em `assets/`.

## Próximo

[02 - Ganhos baratos](02-ganhos-baratos.md): early-abandon, alocações e JSON — e ver que não
bastam (o que prepara o passo 03).
