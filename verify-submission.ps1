<#
.SYNOPSIS
    Dry-run da SUBMISSÃO: sobe a stack a partir da IMAGEM PÚBLICA do Docker Hub
    (submission/docker-compose.yml, com `pull` e sem `build`) e roda o k6.
    É o mesmo artefato que a engine da Rinha vai executar — pega erro de imagem
    privada / nome errado / plataforma ANTES do teste oficial.

.DESCRIPTION
    Diferente do run-test.ps1 (que BUILDA o código local), este faz PULL da imagem
    publicada e usa o compose da branch submission. Projeto docker isolado
    (`rinha-submission`) pra não brigar com a stack de dev.

.PARAMETER SkipSmoke   Pula o smoke, vai direto pra carga.
.PARAMETER SmokeOnly   Só o smoke (valida que responde).
.PARAMETER Down        Derruba a stack no fim (default: deixa de pé).
.PARAMETER NoPull      Não faz pull (usa a imagem já em cache local).

.EXAMPLE
    .\verify-submission.ps1            # pull + up + smoke + carga + nota
    .\verify-submission.ps1 -SmokeOnly # só valida que a imagem pública sobe e responde
#>
[CmdletBinding()]
param(
    [switch]$SkipSmoke,
    [switch]$SmokeOnly,
    [switch]$Down,
    [switch]$NoPull
)

$ErrorActionPreference = 'Stop'

$apiDir      = $PSScriptRoot
$composeFile = Join-Path $apiDir 'submission\docker-compose.yml'
$project     = 'rinha-submission'
$testDir     = (Resolve-Path (Join-Path $apiDir '..\rinha-de-backend-2026\test')).Path
$k6Image     = 'grafana/k6:latest'

function Write-Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "OK  $m"      -ForegroundColor Green }
function Write-Bad($m)  { Write-Host "ERRO $m"     -ForegroundColor Red }

if (-not (Test-Path $composeFile)) { throw "não achei $composeFile" }
# subpasta onde o k6 grava results.json (bug de path do script oficial)
$null = New-Item -ItemType Directory -Force -Path (Join-Path $testDir 'test')

$compose = @('compose', '-f', $composeFile, '-p', $project)

Push-Location $apiDir
try {
    # Evita conflito de porta 9999 com a stack de dev (projeto rinha-fraud).
    Write-Step 'Derrubando a stack de dev (se estiver de pé) pra liberar a :9999'
    docker compose -f (Join-Path $apiDir 'docker-compose.yml') -p rinha-fraud down 2>$null | Out-Host

    if (-not $NoPull) {
        Write-Step 'Pull da imagem pública (Docker Hub)'
        docker @compose pull | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Bad 'pull falhou. A imagem está pública? O nome/tag no compose batem?'
            exit 1
        }
    }

    Write-Step 'Subindo a stack a partir da imagem publicada'
    docker @compose up -d | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'compose up falhou' }

    $nginxId = (docker @compose ps -q nginx).Trim()
    if (-not $nginxId) { throw 'container nginx não encontrado' }
    Write-Ok "nginx container: $nginxId"

    Write-Step 'Aguardando api1 E api2 ficarem prontas'
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        $loaded = (docker @compose logs --no-color 2>$null |
                   Select-String 'reference vectors loaded|loaded prebuilt index').Count
        Write-Host ("  [{0,2}s] instâncias prontas: {1}/2" -f ($i*2), $loaded)
        if ($loaded -ge 2) { $ready = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw 'timeout esperando as APIs (>120s)' }
    Write-Ok 'as duas instâncias estão prontas'
}
finally {
    Pop-Location
}

# --- k6 (compartilhando a rede do nginx) ---
$dockerNet = "container:$nginxId"
$mount     = "${testDir}:/test"

function Invoke-K6($script) {
    docker run --rm --network $dockerNet -v $mount -w /test $k6Image run "/test/$script" 2>&1 | Out-Host
    return $LASTEXITCODE
}

if (-not $SkipSmoke) {
    Write-Step 'SMOKE (valida que a imagem pública responde certo)'
    $code = Invoke-K6 'smoke.js'
    if ($code -ne 0) { Write-Bad "smoke falhou (exit $code). Abortando."; exit 1 }
    Write-Ok 'smoke passou'
}

if (-not $SmokeOnly) {
    Write-Step 'CARGA REAL (test.js, ~2 min — gera results.json)'
    Invoke-K6 'test.js' | Out-Null

    $resultsPath = Join-Path $testDir 'test\results.json'
    Write-Step 'RESULTADO (imagem publicada)'
    if (Test-Path $resultsPath) {
        $r = Get-Content $resultsPath -Raw | ConvertFrom-Json
        $b = $r.scoring.breakdown
        Write-Host ("  p99:          {0}" -f $r.p99)
        Write-Host ("  failure_rate: {0}" -f $r.scoring.failure_rate)
        Write-Host ("  TP/TN/FP/FN:  {0}/{1}/{2}/{3}  (Err: {4})" -f `
            $b.true_positive_detections, $b.true_negative_detections, `
            $b.false_positive_detections, $b.false_negative_detections, $b.http_errors)
        Write-Host ("  p99_score:       {0}" -f $r.scoring.p99_score.value)
        Write-Host ("  detection_score: {0}" -f $r.scoring.detection_score.value)
        $color = if ($r.scoring.final_score -ge 0) { 'Green' } else { 'Red' }
        Write-Host ("  FINAL_SCORE:  {0}" -f $r.scoring.final_score) -ForegroundColor $color
    } else {
        Write-Bad "results.json não foi gerado em $resultsPath"
    }
}

if ($Down) {
    Write-Step 'Derrubando a stack de submissão'
    docker @compose down | Out-Host
} else {
    Write-Host "`n(stack de submissão de pé. Derrubar: docker compose -f '$composeFile' -p $project down)" -ForegroundColor DarkGray
}
