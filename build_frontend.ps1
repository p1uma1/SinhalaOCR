# Build production frontend
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "Node.js/npm not found. Install from https://nodejs.org" -ForegroundColor Red
    exit 1
}

Push-Location frontend
if (-not (Test-Path node_modules)) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    npm install
}
Write-Host "Building production frontend..." -ForegroundColor Cyan
npm run build
Pop-Location

Write-Host "Frontend built to frontend/dist" -ForegroundColor Green
