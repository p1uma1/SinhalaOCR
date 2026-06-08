# Start Sinhala OCR Studio (production API + built frontend)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Virtual env not found. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".\frontend\dist\index.html")) {
    Write-Host "Production frontend not built. Building now..." -ForegroundColor Yellow
    & "$PSScriptRoot\build_frontend.ps1"
}

$hostAddr = if ($env:APP_HOST) { $env:APP_HOST } else { "0.0.0.0" }
$port = if ($env:APP_PORT) { $env:APP_PORT } else { "8000" }

Write-Host "Sinhala OCR Studio → http://127.0.0.1:$port" -ForegroundColor Green
Write-Host "API docs → http://127.0.0.1:$port/api/docs" -ForegroundColor Green

.\.venv\Scripts\python.exe -m uvicorn api.main:app --host $hostAddr --port $port
