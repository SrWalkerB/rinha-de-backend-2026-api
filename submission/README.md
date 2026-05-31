# Branch `submission` — arquivos de deploy

Estes 3 arquivos são o **conteúdo da raiz da branch `submission`** (sem código-fonte,
conforme `docs/br/SUBMISSAO.md`):

| arquivo | o que é |
|---|---|
| `docker-compose.yml` | sobe nginx + api1 + api2 a partir da **imagem pública** (sem `build:`) |
| `nginx.conf` | load balancer round-robin na :9999 |
| `info.json` | metadados da submissão |

A imagem referenciada **já carrega o `index.bin`** (índice IVF pré-construído no build),
então o teste sobe sem dataset e sem fonte.

## Fluxo completo de submissão

### 1. Publicar a imagem (uma vez por versão)

```powershell
.\publish.ps1 -User SEU_USUARIO_DOCKERHUB
# builda com index.bin embutido (linux/amd64) e faz push pro registry público
```

Depois, ajuste a linha `image:` em `submission/docker-compose.yml` pro nome que você publicou
(o default é `srwalkerb/rinha-fraud:1.0`).

### 2. Branches do repositório

O repo já vem com a estrutura pronta:
- **`main`** — código-fonte completo.
- **`submission`** — só os 3 arquivos acima na raiz.

Para **regerar/atualizar** a branch `submission` depois de mexer nestes arquivos:

```powershell
git checkout submission
Copy-Item submission\docker-compose.yml, submission\nginx.conf, submission\info.json . -Force
git add docker-compose.yml nginx.conf info.json
git commit -m "update submission deploy files"
git checkout main
```

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
