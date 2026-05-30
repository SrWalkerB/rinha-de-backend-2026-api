<#
.SYNOPSIS
    Gerador de carga leve para a API LOCAL (go run, :8080) — usado para perfilar com pprof.

.DESCRIPTION
    O run-test.ps1 mira a stack Docker (nginx :9999). Para o passo 01 (profiling) a API roda
    local via `go run .` em :8080, então este script martela esse endpoint com N workers em
    paralelo por alguns segundos, mantendo a CPU ocupada enquanto o pprof captura o profile.

    Cada request dispara a busca O(3M) (~25ms de CPU), então mesmo poucos workers saturam um
    core. Rode isto numa janela enquanto captura o profile noutra.

.PARAMETER Url
    Endpoint. Default http://localhost:8080/fraud-score

.PARAMETER Seconds
    Duração. Default 40 (cobre uma captura de 30s com folga).

.PARAMETER Workers
    Requests concorrentes. Default 8.

.EXAMPLE
    .\hammer.ps1                       # 8 workers, 40s, contra :8080
    .\hammer.ps1 -Seconds 60 -Workers 12
#>
[CmdletBinding()]
param(
    [string]$Url = 'http://localhost:8088/fraud-score',
    [int]$Seconds = 40,
    [int]$Workers = 8
)

$ErrorActionPreference = 'Stop'

# Payload de exemplo (primeira transação do arquivo de exemplos do desafio).
$payloadFile = Join-Path $PSScriptRoot 'resources\example-payloads.json'
$body = ((Get-Content $payloadFile -Raw | ConvertFrom-Json)[0] | ConvertTo-Json -Depth 10 -Compress)

Write-Host "Martelando $Url por ${Seconds}s com $Workers workers..." -ForegroundColor Cyan
Write-Host "(deixe rodando e capture o profile noutra janela; confira 1 core ~100%)" -ForegroundColor DarkGray

$deadline = (Get-Date).AddSeconds($Seconds)

$counts = 1..$Workers | ForEach-Object -Parallel {
    $end = $using:deadline
    $u   = $using:Url
    $b   = $using:body
    $n = 0
    while ((Get-Date) -lt $end) {
        try {
            Invoke-RestMethod -Uri $u -Method Post -Body $b -ContentType 'application/json' -TimeoutSec 5 | Out-Null
            $n++
        } catch { }
    }
    $n
} -ThrottleLimit $Workers

$total = ($counts | Measure-Object -Sum).Sum
$rps   = [math]::Round($total / $Seconds, 1)
Write-Host "Feito. $total requests em ${Seconds}s (~$rps req/s)." -ForegroundColor Green
