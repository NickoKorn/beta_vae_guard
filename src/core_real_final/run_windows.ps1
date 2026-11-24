# Start from PowerShell: .\src\core_real_final\run_windows.ps1
# Forward main.py options after -TrainingArgs, for example:
# .\src\core_real_final\run_windows.ps1 -Single -TrainingArgs --group weak_signal_lowvar --variant base --split-seed 1 --train-seed 1
[CmdletBinding()]
param(
    [string]$Output = "",
    [string]$Config = "",
    [switch]$Single,
    [int]$Workers = 1,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainingArgs
)

$ErrorActionPreference = "Stop"
$CoreDir = $PSScriptRoot
$ProjectDir = Split-Path (Split-Path $CoreDir -Parent) -Parent
Set-Location $ProjectDir
if (-not $Output) { $Output = Join-Path $CoreDir "results\windows" }
if (-not $Config) { $Config = Join-Path $CoreDir "guarded_varix.json" }
$Config = (Resolve-Path $Config).Path
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Output = (Resolve-Path $Output).Path

$VenvDir = Join-Path $CoreDir ".venv-windows"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $CoreDir "requirements-windows.txt"
$Autoencodix = "https://github.com/jan-forest/autoencodix_package/archive/b405aa3ec037552370805fb3c9f0324c3baa6895.zip"
$Marker = Join-Path $VenvDir "guarded-varix-dependencies.txt"
$DependencyHash = (Get-FileHash $Requirements -Algorithm SHA256).Hash + $Autoencodix

if (-not (Test-Path $Python)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python launcher not found. Install Python 3.11 (64-bit) and retry."
    }
    & py -3.11 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Creating the Python 3.11 environment failed." }
}

# Activating modifies only this PowerShell process; use the explicit interpreter as well.
& (Join-Path $VenvDir "Scripts\Activate.ps1")
if (-not (Test-Path $Marker) -or (Get-Content $Marker -Raw) -ne $DependencyHash) {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Upgrading pip failed." }
    & $Python -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "Installing Windows dependencies failed." }
    # The pinned package declares optional profiling dependencies unavailable on Windows.
    & $Python -m pip install --no-deps $Autoencodix
    if ($LASTEXITCODE -ne 0) { throw "Installing AUTOENCODIX failed." }
    [System.IO.File]::WriteAllText($Marker, $DependencyHash)
}

$env:PYTHONHASHSEED = "1"
$env:PYTHONUTF8 = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
& $Python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA GPU is unavailable. Check the NVIDIA driver and the PyTorch CUDA installation." }

# Save the effective configuration alongside results, preserving any previous copy.
$ConfigData = Get-Content -Raw -Path $Config | ConvertFrom-Json
$ConfigData.protocol.collapse.lucas_epsilon_nats = 0.01
$RunConfig = Join-Path $Output ("config-eps001-" + (Get-Date -Format "yyyyMMdd-HHmmss-fffffff") + ".json")
$Json = $ConfigData | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText($RunConfig, $Json, [System.Text.UTF8Encoding]::new($false))

if ($Workers -lt 1) { throw "Workers must be at least 1." }
if ($Workers -gt 1 -and $Single) { throw "Use either -Single or multiple workers." }
if ($Workers -gt 1 -and $TrainingArgs) { throw "Edit the config for a parallel grid; extra TrainingArgs are only supported by the serial runner." }
if ($Workers -gt 1) {
    $Arguments = @("-u", "-m", "src.core_real_final.parallel_grid", $Output, "--config", $RunConfig, "--workers", "$Workers")
} else {
    $Arguments = @("-u", "-m", "src.core_real_final.main", $Output, "--config", $RunConfig)
    if (-not $Single) { $Arguments += "--grid" }
}
if ($TrainingArgs) { $Arguments += $TrainingArgs }
Write-Host "Output: $Output"
Write-Host "Config: $RunConfig"
Write-Host "Lucas epsilon: 0.01 Nat"
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE. Existing results are in $Output." }
