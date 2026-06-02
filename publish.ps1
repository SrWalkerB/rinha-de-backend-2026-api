<#
.SYNOPSIS
    Builda a imagem (com o index.bin embutido) e faz push pra um registry PÚBLICO.
    A branch `submission` aponta pra essa imagem (não pode ter código-fonte).

.PARAMETER User
    Seu usuário do Docker Hub (minúsculo). Ex.: srwalkerb

.PARAMETER Tag
    Tag da imagem (default: 2.0). Use uma TAG NOVA a cada push pra evitar cache da
    engine e CASAR com submission/docker-compose.yml.

.PARAMETER SkipPush
    Só builda (não faz push) — pra testar localmente antes.

.EXAMPLE
    .\publish.ps1 -User srwalkerb -Tag 2.0
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$User,
    [string]$Tag = '2.0',
    [switch]$SkipPush
)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $image = "$($User.ToLower())/rinha-fraud:$Tag"

    if (-not (Test-Path '.\resources\references.json.gz')) {
        throw "Falta resources\references.json.gz — necessário no build pra cozinhar o index.bin."
    }

    Write-Host "`n=== Build $image (linux/amd64; IVF nlist baked = 1024) ===" -ForegroundColor Cyan
    # --platform garante linux/amd64 (host de avaliação é Mac Mini Haswell amd64).
    # O índice IVF (nlist=1024) é cozido pelo buildindex dentro do Dockerfile.
    docker build --platform linux/amd64 -t $image .
    if ($LASTEXITCODE -ne 0) { throw 'docker build falhou' }

    if ($SkipPush) {
        Write-Host "`nOK build (push pulado). Imagem local: $image" -ForegroundColor Green
        return
    }

    Write-Host "`n=== Push $image ===" -ForegroundColor Cyan
    Write-Host "(se pedir login: docker login)" -ForegroundColor DarkGray
    docker push $image
    if ($LASTEXITCODE -ne 0) { throw 'docker push falhou — rode `docker login` e tente de novo' }

    Write-Host "`n=== PRONTO ===" -ForegroundColor Green
    Write-Host "Imagem pública: $image"
    Write-Host "Agora atualize submission\docker-compose.yml com essa imagem e suba a branch submission."
    Write-Host "Veja submission\README.md pros comandos de git." -ForegroundColor DarkGray
}
finally {
    Pop-Location
}
