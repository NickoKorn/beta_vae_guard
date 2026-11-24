# Start from the repository root:
# powershell -ExecutionPolicy Bypass -File .\windows\run_windows_parallel4.ps1
[CmdletBinding()]
param(
    [string]$Output = "",
    [string]$Config = ""
)

$ProjectDir = Split-Path $PSScriptRoot -Parent
$CoreDir = Join-Path $ProjectDir "src\core_real_final"
if (-not $Output) { $Output = Join-Path $CoreDir "results\windows_parallel4" }
& (Join-Path $PSScriptRoot "run_windows.ps1") -Output $Output -Config $Config -Workers 4
if (-not $?) { exit 1 }
