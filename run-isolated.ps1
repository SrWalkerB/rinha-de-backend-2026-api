<#
  Temp / uncommitted. Isolated docker measurement for Caminho A (quantize x10000).
  Builds NOTHING (use `docker build -t rinha-fraud:local .` first). Runs a self-
  contained stack (own network + caps mirroring the eval) and the k6 load test,
  WITHOUT touching the rinha-submission containers. Tears down at the end.
#>
[CmdletBinding()]
param([int]$NProbe = 12, [string]$Scorer = 'knn')

$ErrorActionPreference = 'Stop'
$apiDir  = $PSScriptRoot
$testDir = (Resolve-Path (Join-Path $PSScriptRoot '..\rinha-de-backend-2026\test')).Path
$img     = 'rinha-fraud:local'
$net     = 'rinhalocal'
$nginxConf = Join-Path $apiDir 'nginx.conf'

# results.json path bug in the official handleSummary: it writes test/results.json
# relative to the k6 workdir /test, so ensure that subdir exists on the host.
$null = New-Item -ItemType Directory -Force -Path (Join-Path $testDir 'test')

function Cleanup {
    Write-Host "`n=== teardown ===" -ForegroundColor DarkGray
    docker rm -f rl-api1 rl-api2 rl-nginx 2>$null | Out-Null
    docker network rm $net 2>$null | Out-Null
}

# Clean any leftovers from a previous run
docker rm -f rl-api1 rl-api2 rl-nginx 2>$null | Out-Null
docker network rm $net 2>$null | Out-Null

try {
    Write-Host "=== network ===" -ForegroundColor Cyan
    docker network create $net | Out-Null

    Write-Host "=== api1 + api2 (cpus 0.45 / 155m, GOMAXPROCS=1, SCORER=$Scorer, KNN_NPROBE=$NProbe) ===" -ForegroundColor Cyan
    foreach ($n in 'api1','api2') {
        docker run -d --name "rl-$n" --network $net --network-alias $n `
            --cpus 0.45 --memory 155m `
            -e GOMAXPROCS=1 -e GOMEMLIMIT=150MiB -e GOGC=off -e SCORER=$Scorer -e KNN_NPROBE=$NProbe -e ADDR=:8080 `
            $img | Out-Null
    }

    Write-Host "=== nginx (cpus 0.10 / 40m, listen 9999) ===" -ForegroundColor Cyan
    docker run -d --name rl-nginx --network $net `
        --cpus 0.10 --memory 40m `
        -v "${nginxConf}:/etc/nginx/nginx.conf:ro" `
        nginx:1.27-alpine | Out-Null

    Write-Host "=== aguardando api1 E api2 (loaded prebuilt index) ===" -ForegroundColor Cyan
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        $loaded = (docker logs rl-api1 2>&1 | Select-String 'ready:').Count `
                + (docker logs rl-api2 2>&1 | Select-String 'ready:').Count
        Write-Host ("  [{0,2}s] prontas: {1}/2" -f ($i*2), $loaded)
        if ($loaded -ge 2) { $ready = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) {
        Write-Host "TIMEOUT — logs api1:" -ForegroundColor Red
        docker logs rl-api1 2>&1 | Out-Host
        throw 'apis nao ficaram prontas'
    }
    Write-Host "OK as duas prontas" -ForegroundColor Green

    Write-Host "=== k6 carga (test.js, ~2 min) ===" -ForegroundColor Cyan
    docker run --rm --network "container:rl-nginx" -v "${testDir}:/test" -w /test grafana/k6:latest run /test/test.js 2>&1 | Out-Host

    $resultsPath = Join-Path $testDir 'test\results.json'
    Write-Host "`n=== RESULTADO ===" -ForegroundColor Cyan
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
        Write-Host ("  FINAL_SCORE:  {0}" -f $r.scoring.final_score) -ForegroundColor Green
    } else {
        Write-Host "results.json NAO gerado em $resultsPath" -ForegroundColor Red
    }
}
finally {
    Cleanup
}
