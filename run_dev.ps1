# Development: Vite frontend (5173) + FastAPI backend (8000)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Starting API on http://127.0.0.1:8000" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; .\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"

Start-Sleep -Seconds 2

Write-Host "Starting frontend on http://127.0.0.1:5173" -ForegroundColor Cyan
Push-Location frontend
if (-not (Test-Path node_modules)) { npm install }
npm run dev
