# Start from the repository root:
# powershell -ExecutionPolicy Bypass -File .\src\core_real_final\run_windows_parallel4.ps1
[CmdletBinding()]
param(
    [string]$Output = "",
    [string]$Config = ""
)

if (-not $Output) { $Output = Join-Path $PSScriptRoot "results\windows_parallel4" }
& (Join-Path $PSScriptRoot "run_windows.ps1") -Output $Output -Config $Config -Workers 4
if (-not $?) { exit 1 }
