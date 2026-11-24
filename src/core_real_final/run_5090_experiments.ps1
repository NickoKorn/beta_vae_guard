# Start from the repository root:
# powershell -ExecutionPolicy Bypass -File .\src\core_real_final\run_5090_experiments.ps1
[CmdletBinding()]
param(
    [string]$Output = 'D:\Nickos Exp\rtx5090_eps01',
    [ValidateSet('all', 'tcga_baselines', 'gudhi')]
    [string]$Suite = 'all',
    [int]$Workers = 4
)

$ErrorActionPreference = 'Stop'
$CoreDir = $PSScriptRoot
$ProjectDir = Split-Path (Split-Path $CoreDir -Parent) -Parent
$VenvDir = Join-Path $CoreDir '.venv-windows'
$Python = Join-Path $VenvDir 'Scripts\python.exe'
$Requirements = Join-Path $CoreDir 'requirements-windows.txt'
$Autoencodix = 'https://github.com/jan-forest/autoencodix_package/archive/b405aa3ec037552370805fb3c9f0324c3baa6895.zip'
$Marker = Join-Path $VenvDir 'guarded-varix-dependencies.txt'

if ($Workers -lt 1) { throw 'Workers must be at least 1.' }
if (-not (Test-Path $Python)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Python launcher not found. Install Python 3.11 (64-bit) and retry.'
    }
    & py -3.11 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw 'Creating the Python 3.11 environment failed.' }
}

& (Join-Path $VenvDir 'Scripts\Activate.ps1')
$DependencyHash = (Get-FileHash $Requirements -Algorithm SHA256).Hash + $Autoencodix
if (-not (Test-Path $Marker) -or (Get-Content $Marker -Raw) -ne $DependencyHash) {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Upgrading pip failed.' }
    & $Python -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw 'Installing Windows dependencies failed.' }
    & $Python -m pip install --no-deps $Autoencodix
    if ($LASTEXITCODE -ne 0) { throw 'Installing AUTOENCODIX failed.' }
    [System.IO.File]::WriteAllText($Marker, $DependencyHash)
}

$env:PYTHONHASHSEED = '1'
$env:PYTHONUTF8 = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $Python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw 'CUDA GPU is unavailable. Check the NVIDIA driver and PyTorch CUDA installation.' }

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Output = (Resolve-Path $Output).Path
Set-Location $ProjectDir
Write-Host "Output: $Output"
Write-Host "Suite: $Suite, workers: $Workers"
Write-Host 'Starting guarded Varix experiments with Lucas epsilon 0.1 Nat.'
& $Python -u -m src.core_real_final.run_5090_experiments $Output --suite $Suite --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "Experiment suite failed with exit code $LASTEXITCODE. See worker_logs in $Output." }
