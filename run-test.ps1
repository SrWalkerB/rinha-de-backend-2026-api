<#
.SYNOPSIS
    Sobe a stack da API (LB próprio L4 + api1 + api2), espera ficar pronta e roda o k6.

.DESCRIPTION
    Um comando = build + up + wait-ready + smoke + carga + resultado.
    Resolve sozinho o nome do container do load balancer (lb próprio, fallback nginx;
    sem hardcode) e amarra o k6 na rede dele -- contorna o `network_mode: host`
    quebrado do Docker Desktop Win/Mac.

.PARAMETER SkipSmoke
    Pula o smoke e vai direto pra carga real.

.PARAMETER SmokeOnly
    Roda só o smoke (validação rápida), não roda a carga.

.PARAMETER Down
    Derruba a stack no final (default: deixa de pé pra próxima rodada ser rápida).

.PARAMETER NoBuild
    Sobe sem rebuildar a imagem (mais rápido se o código não mudou).

.EXAMPLE
    .\run-test.ps1                 # build + up + smoke + carga + resultado
    .\run-test.ps1 -SmokeOnly      # só valida que responde
    .\run-test.ps1 -SkipSmoke -Down  # direto na carga, derruba no fim
#>
[CmdletBinding()]
param(
    [switch]$SkipSmoke,
    [switch]$SmokeOnly,
    [switch]$Down,
    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'

# --- Caminhos (relativos ao próprio script, então funciona de qualquer lugar) ---
$apiDir  = $PSScriptRoot
$testDir = (Resolve-Path (Join-Path $PSScriptRoot '..\rinha-de-backend-2026\test')).Path
$k6Image = 'grafana/k6:latest'

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "OK  $msg" -ForegroundColor Green }
function Write-Bad($msg)  { Write-Host "ERRO $msg" -ForegroundColor Red }

# Garante a subpasta onde o k6 grava o results.json (bug de path do script oficial)
$null = New-Item -ItemType Directory -Force -Path (Join-Path $testDir 'test')

# --- 1. Sobe a API ---------------------------------------------------------
Write-Step 'Subindo a stack da API (LB L4 próprio + api1 + api2)'
Push-Location $apiDir
try {
    if ($NoBuild) {
        docker compose up -d | Out-Host
    } else {
        docker compose up -d --build | Out-Host
    }
    if ($LASTEXITCODE -ne 0) { throw 'docker compose up falhou' }

    # Descobre o container do load balancer (lb próprio OU nginx) dinamicamente.
    $nginxId = (docker compose ps -q lb 2>$null).Trim()
    if (-not $nginxId) { $nginxId = (docker compose ps -q nginx 2>$null).Trim() }
    if (-not $nginxId) { throw 'container do load balancer (lb/nginx) não encontrado' }
    Write-Ok "load balancer container: $nginxId"

    # --- 2. Espera AMBAS as instâncias carregarem os 3M vetores ------------
    Write-Step 'Aguardando api1 E api2 ficarem prontas (carregando 3M vetores)'
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        # Cada instância loga UMA destas ao ficar pronta: "reference vectors
        # loaded" (fallback que constrói no startup) ou "loaded prebuilt index"
        # (índice pré-construído, passo 05).
        $loaded = (docker compose logs --no-color 2>$null |
                   Select-String 'reference vectors loaded|loaded prebuilt index').Count
        Write-Host ("  [{0,2}s] instâncias prontas: {1}/2" -f ($i*2), $loaded)
        if ($loaded -ge 2) { $ready = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw 'timeout esperando as APIs ficarem prontas (>180s)' }
    Write-Ok 'as duas instâncias estão prontas'
}
finally {
    Pop-Location
}

# --- 3. Roda o k6 (compartilhando a rede do load balancer) -----------------
$dockerNet = "container:$nginxId"
$mount     = "${testDir}:/test"

function Invoke-K6($script) {
    # Saída do docker vai direto pro host (Out-Host), então o ÚNICO valor que
    # sai pela pipeline da função é o exit code. Sem isso, $code captura todo o
    # texto do k6 e a comparação de exit code quebra.
    docker run --rm --network $dockerNet -v $mount -w /test $k6Image run "/test/$script" 2>&1 | Out-Host
    return $LASTEXITCODE
}

if (-not $SkipSmoke) {
    Write-Step 'SMOKE (valida que responde — checks rate==1.0)'
    $code = Invoke-K6 'smoke.js'
    if ($code -ne 0) {
        Write-Bad "smoke falhou (exit $code). API não está respondendo certo. Abortando."
        exit 1
    }
    Write-Ok 'smoke passou'
}

if (-not $SmokeOnly) {
    Write-Step 'CARGA REAL (test.js, ~2 min — gera results.json)'
    # Não aborta se thresholds falharem: ainda queremos ver o results.json.
    Invoke-K6 'test.js' | Out-Null

    # --- 4. Mostra o resultado --------------------------------------------
    $resultsPath = Join-Path $testDir 'test\results.json'
    Write-Step 'RESULTADO'
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
        Write-Host "`n  (json completo: $resultsPath)"
    } else {
        Write-Bad "results.json não foi gerado em $resultsPath"
    }
}

# --- 5. Derruba (opcional) -------------------------------------------------
if ($Down) {
    Write-Step 'Derrubando a stack'
    Push-Location $apiDir
    try { docker compose down | Out-Host } finally { Pop-Location }
} else {
    Write-Host "`n(API segue de pé. Para derrubar: cd '$apiDir'; docker compose down)" -ForegroundColor DarkGray
}
