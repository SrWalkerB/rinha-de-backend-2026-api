# Branch `submission` — arquivos de deploy

Conteúdo da raiz da branch `submission` (sem código-fonte, conforme `docs/br/SUBMISSAO.md`):

| arquivo | o que é |
|---|---|
| `docker-compose.yml` | sobe **lb (L4 próprio) + api1 + api2** a partir da **imagem pública** (sem `build:`) |
| `info.json` | metadados da submissão (stack = `["go"]`) |

> `nginx.conf` é **legado** — a stack atual NÃO usa nginx; o load balancer é o nosso
> binário Go L4 (`/app/lb`, mesma imagem, entrypoint próprio). Pode ser removido da
> branch `submission`. Mantido só como referência do fallback de revert pra `2.2`.

A imagem referenciada **já carrega o `index.bin`** (índice IVF pré-construído no build),
então o teste sobe sem dataset e sem fonte.

## Fluxo completo de submissão

### 1. Publicar a imagem (uma vez por versão)

```powershell
.\publish.ps1 -User SEU_USUARIO_DOCKERHUB -Tag 3.0
# builda com index.bin embutido (linux/amd64) e faz push pro registry público
```

Depois, ajuste a linha `image:` em `submission/docker-compose.yml` pro nome que você publicou
(atual: `srwalkerb/rinha-fraud:3.0`). Use TAG NOVA a cada versão (busta o cache da engine de
avaliação e casa com o compose).

### 2. Branches do repositório

O repo já vem com a estrutura pronta:
- **`main`** — código-fonte completo.
- **`submission`** — só os 3 arquivos acima na raiz.

Para **regerar/atualizar** a branch `submission` depois de mexer nestes arquivos:

```powershell
git checkout submission
Copy-Item submission\docker-compose.yml, submission\info.json . -Force
git rm --ignore-unmatch nginx.conf            # legado: stack agora é go-only (L4 LB próprio)
git add docker-compose.yml info.json
git commit -m "submission: image 3.0 (SIMD SoA int16 + LB L4 próprio), stack go-only"
git checkout main
```

> **Revert pro provado (4056):** se a prévia da `3.0` regredir vs 4056, volte a branch
> `submission` pro commit `88a459f` (image `2.2` + nginx 0.30/api 0.35×2) ANTES de 06-05 —
> o teste final usa `origin/submission` na data, então um preview ruim é recuperável.

### 3. Subir pro GitHub (repo PÚBLICO)

```powershell
git remote add origin https://github.com/SrWalkerB/rinha-de-backend-2026-api.git
git push -u origin main
git push -u origin submission
```

### 4. PR de inscrição

Você faz (no repo `zanfranceschi/rinha-de-backend-2026`): adicionar
`participants/SrWalkerB.json`:

```json
[{ "id": "srwalkerb-go", "repo": "https://github.com/SrWalkerB/rinha-de-backend-2026-api" }]
```
