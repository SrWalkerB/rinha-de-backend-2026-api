<#
.SYNOPSIS
  Sweep da CPU do LB (L4 próprio) 0.10->0.30, api=(1-lb)/2 cada. Mede p99/final/Err/E
  por config E o BALANCE api1/api2 (soma de CPU% amostrada durante a carga) — pra decidir
  se o RR-por-conexão do L4 skewa (gate do Exp B per-request).

  ATENÇÃO: p99 LOCAL = alto ruído + NÃO reproduz o starvation do LB do Mac (lb0.10 local
  87ms vs Mac 194ms). Use o p99 só pra REGIME; o BALANCE é o sinal local confiável.
  Stack já deve estar de pé e PRONTA (run-test.ps1 ou docker compose up -d --build).
#>
[CmdletBinding()]
param([int]$Settle = 4)

$ErrorActionPreference = 'Stop'
$api     = $PSScriptRoot
$testDir = (Resolve-Path (Join-Path $api '..\rinha-de-backend-2026\test')).Path
$lb='rinha-fraud-lb-1'; $a1='rinha-fraud-api1-1'; $a2='rinha-fraud-api2-1'
$k6='grafana/k6:latest'; $net="container:$lb"; $mount="${testDir}:/test"
$out = Join-Path $api 'sweep-results.txt'
"=== LB CPU sweep $(Get-Date -Format s) (app = centroid-SIMD dev build) ===" | Out-File $out

$configs = @(
  @{lb='0.10'; api='0.45'},
  @{lb='0.15'; api='0.425'},
  @{lb='0.20'; api='0.40'},
  @{lb='0.25'; api='0.375'},
  @{lb='0.30'; api='0.35'}
)

foreach ($c in $configs) {
  docker update --cpus=$($c.lb)  $lb | Out-Null
  docker update --cpus=$($c.api) $a1 | Out-Null
  docker update --cpus=$($c.api) $a2 | Out-Null
  Start-Sleep -Seconds $Settle

  # Balance sampler (parallel): sum CPU% of api1/api2 over ~the first 40s of load.
  $job = Start-Job -ScriptBlock {
    param($a1,$a2)
    $s1=0.0; $s2=0.0; $n=0
    for ($i=0; $i -lt 20; $i++) {
      $st = docker stats --no-stream --format '{{.Name}} {{.CPUPerc}}' $a1 $a2 2>$null
      foreach ($line in $st) {
        $p = $line -split '\s+'
        if ($p.Count -lt 2) { continue }
        $cpu = [double](($p[1] -replace '%','').Trim())
        if ($p[0] -eq $a1) { $s1 += $cpu } elseif ($p[0] -eq $a2) { $s2 += $cpu }
      }
      $n++
    }
    "$s1|$s2|$n"
  } -ArgumentList $a1,$a2

  # Real load (blocks ~2 min). Overwrites results.json.
  docker run --rm --network $net -v $mount -w /test $k6 run /test/test.js 2>&1 | Out-Null

  $bal = (Receive-Job -Job $job -Wait); Remove-Job $job
  $parts = $bal -split '\|'
  $c1=[double]$parts[0]; $c2=[double]$parts[1]
  $skew = if (($c1+$c2) -gt 0) { [math]::Round([math]::Max($c1,$c2)/[math]::Max(($c1+$c2)/2,0.01),2) } else { 0 }

  try {
    $r = Get-Content (Join-Path $testDir 'test\results.json') -Raw | ConvertFrom-Json
    $b = $r.scoring.breakdown
    $row = ("lb={0}/api={1}x2 | p99={2} final={3} Err={4} FP={5} FN={6} | bal a1/a2={7:N0}/{8:N0} skew={9}x" -f `
      $c.lb,$c.api,$r.p99,$r.scoring.final_score,$b.http_errors,$b.false_positive_detections,$b.false_negative_detections,$c1,$c2,$skew)
  } catch {
    $row = ("lb={0}/api={1}x2 | ERRO parse results.json: {2}" -f $c.lb,$c.api,$_.Exception.Message)
  }
  $row | Tee-Object -FilePath $out -Append | Out-Host
}

# Restore the dev default (lb 0.10 / api 0.45).
docker update --cpus=0.10 $lb | Out-Null
docker update --cpus=0.45 $a1 | Out-Null
docker update --cpus=0.45 $a2 | Out-Null
"DONE $(Get-Date -Format s)" | Tee-Object -FilePath $out -Append | Out-Host
