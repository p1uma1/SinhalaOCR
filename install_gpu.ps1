# Install CUDA-enabled PyTorch for NVIDIA RTX 5090 (Blackwell, sm_120).
# Requires CUDA 12.8+ — older cu124 builds will NOT work on this GPU.
#
# Usage (from project root):
#   .\.venv\Scripts\Activate.ps1
#   .\install_gpu.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Virtual env not found. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

Write-Host "Removing CPU-only PyTorch (if installed)..." -ForegroundColor Cyan
& .\.venv\Scripts\pip.exe uninstall -y torch torchvision torchaudio *> $null

Write-Host "Installing PyTorch with CUDA 12.8 (RTX 5090 / Blackwell)..." -ForegroundColor Cyan
& .\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "`nVerifying GPU..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.randn(1024, 1024, device='cuda')
    y = x @ x
    print(f'GPU matmul OK: {y.shape}')
else:
    print('FAILED - trying nightly cu128 build...')
    raise SystemExit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "Stable cu128 failed. Installing nightly cu128..." -ForegroundColor Yellow
    & .\.venv\Scripts\pip.exe uninstall -y torch torchvision torchaudio *> $null
    & .\.venv\Scripts\pip.exe install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    & .\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print('Nightly OK:', torch.cuda.get_device_name(0))"
}

Write-Host "`nGPU PyTorch ready. Run: python verify_setup.py" -ForegroundColor Green
