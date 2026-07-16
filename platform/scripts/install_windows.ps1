#Requires -Version 5.1
<#
.SYNOPSIS
    Simple setup for the community-compute worker client on Windows.

.DESCRIPTION
    Creates a Python virtual environment, installs dependencies, saves your
    connection settings into a generated launcher script (run.bat), and
    optionally registers a Scheduled Task so the worker starts automatically
    at logon and restarts if it crashes.

    Mirrors platform/scripts/install_linux.sh -- same flags, same
    generated-launcher approach, same "everything needed is under
    platform/worker/" install target. See platform/docs/WORKER.md
    for how to get a server URL and API key.

.PARAMETER Server
    Platform server base URL, e.g. https://compute.example.org

.PARAMETER EngineBin
    Path to the compiled chess engine UCI executable (chess.exe).

.PARAMETER ApiKey
    Your per-account API key (from the server's /accounts endpoints).
    Leave unset to use -RegistrationSecret instead.

.PARAMETER RegistrationSecret
    Legacy shared registration secret, only if the server operator enabled
    one. Prefer -ApiKey.

.PARAMETER Threads
    Number of concurrent self-play threads. Default 1.

.PARAMETER MaxCpuPercent
    Optional soft CPU cap (see resource_limits.py).

.PARAMETER MaxMemoryMb
    Optional hard memory cap (see resource_limits.py).

.PARAMETER InstallService
    Register a Scheduled Task that starts the worker at logon.

.PARAMETER NoService
    Skip Scheduled Task registration (default if neither switch is given
    and -NonInteractive is set).

.PARAMETER NonInteractive
    Fail on missing required values instead of prompting (for scripted
    installs).

.EXAMPLE
    .\install_windows.ps1 -Server https://compute.example.org `
        -EngineBin C:\chess\chess.exe -ApiKey cek_xxxxxxxx -Threads 4 -InstallService
#>
[CmdletBinding()]
param(
    [string]$Server,
    [string]$EngineBin,
    [string]$ApiKey,
    [string]$RegistrationSecret,
    [int]$Threads,
    [double]$MaxCpuPercent,
    [double]$MaxMemoryMb,
    [switch]$InstallService,
    [switch]$NoService,
    [switch]$NonInteractive
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkerDir = Resolve-Path (Join-Path $ScriptDir '..\worker')

function Read-OrPrompt {
    param([string]$Value, [string]$Question, [string]$Default = '')
    if ($Value) { return $Value }
    if ($NonInteractive) {
        if (-not $Default) {
            throw "missing required value: $Question (pass it as a parameter in -NonInteractive mode)"
        }
        return $Default
    }
    $prompt = if ($Default) { "$Question [$Default]" } else { $Question }
    $ans = Read-Host $prompt
    if (-not $ans) { $ans = $Default }
    return $ans
}

Write-Host "=== Morph Community Compute -- worker installer (Windows) ===" -ForegroundColor Cyan
Write-Host "install directory: $WorkerDir"
Write-Host ""

$Server = Read-OrPrompt -Value $Server -Question 'Platform server URL (e.g. https://compute.example.org)'
$EngineBin = Read-OrPrompt -Value $EngineBin -Question 'Path to compiled chess engine UCI executable'
if (-not $ApiKey -and -not $RegistrationSecret -and -not $NonInteractive) {
    $ApiKey = Read-Host 'Your API key (from the server''s /accounts endpoints -- leave blank to use a registration secret instead)'
}
if (-not $Threads) {
    $ThreadsStr = Read-OrPrompt -Value $null -Question 'Number of concurrent self-play threads' -Default '1'
    $Threads = [int]$ThreadsStr
}

if (-not $Server) { throw 'server URL is required' }
if (-not (Test-Path $EngineBin)) {
    Write-Warning "$EngineBin does not look like an existing file -- continuing anyway"
}

# --- find a Python interpreter (py launcher preferred, falls back to python) ---
$PythonCmd = $null
foreach ($candidate in @('py', 'python', 'python3')) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $PythonCmd = $candidate
        break
    }
}
if (-not $PythonCmd) {
    throw 'Python was not found on PATH. Install Python 3.9+ from https://python.org (check ' + `
          '"Add python.exe to PATH" during setup) and re-run this installer.'
}
$pyVersionOutput = & $PythonCmd --version 2>&1
Write-Host "found $PythonCmd ($pyVersionOutput)"

# --- venv ---
$VenvDir = Join-Path $WorkerDir 'venv'
Write-Host "--- creating virtual environment at $VenvDir ---"
if (Test-Path $VenvDir) {
    Write-Host "venv already exists, reusing it (delete $VenvDir to force a clean rebuild)"
} else {
    & $PythonCmd -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw 'python -m venv failed' }
}

$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
& $VenvPython -m pip install --quiet --upgrade pip
& $VenvPython -m pip install --quiet -r (Join-Path $WorkerDir 'requirements.txt')
Write-Host "dependencies installed"

# --- generated launcher (run.bat) ---
$RunBat = Join-Path $WorkerDir 'run.bat'
$argLines = @()
$argLines += "`"$VenvPython`" run_platform_worker.py ^"
$argLines += "    --server `"$Server`" ^"
$argLines += "    --engine-bin `"$EngineBin`" ^"
if ($ApiKey) { $argLines += "    --api-key `"$ApiKey`" ^" }
if ($RegistrationSecret) { $argLines += "    --registration-secret `"$RegistrationSecret`" ^" }
$argLines += "    --threads $Threads ^"
if ($MaxCpuPercent) { $argLines += "    --max-cpu-percent $MaxCpuPercent ^" }
if ($MaxMemoryMb) { $argLines += "    --max-memory-mb $MaxMemoryMb ^" }
$argLines += "    %*"

$batContent = @(
    "@echo off",
    "REM Generated by install_windows.ps1 on $(Get-Date -Format o). Edit freely, or re-run",
    "REM install_windows.ps1 to regenerate.",
    "cd /d `"$WorkerDir`"",
    ($argLines -join "`r`n")
) -join "`r`n"
Set-Content -Path $RunBat -Value $batContent -Encoding ASCII
Write-Host "wrote launcher: $RunBat"

# --- optional Scheduled Task ---
$doService = $false
if ($InstallService) { $doService = $true }
elseif ($NoService) { $doService = $false }
elseif (-not $NonInteractive) {
    $yn = Read-Host 'Register a Scheduled Task to auto-start the worker at logon? [y/N]'
    $doService = $yn -match '^[Yy]'
}

if ($doService) {
    $taskName = 'ChessComputeWorker'
    try {
        $action = New-ScheduledTaskAction -Execute $RunBat
        $trigger = New-ScheduledTaskTrigger -AtLogOn
        $settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
            -ExecutionTimeLimit (New-TimeSpan -Days 0) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
            -Description 'Morph Community Compute worker' -Force | Out-Null
        Write-Host "registered Scheduled Task: $taskName (starts at logon, auto-restarts on failure)"
        Write-Host "start it now with: Start-ScheduledTask -TaskName $taskName"
        Write-Host "view/manage it in Task Scheduler, or: Get-ScheduledTask -TaskName $taskName"
    } catch {
        Write-Warning "Scheduled Task registration failed ($_) -- you may need to run this script as " + `
                       "Administrator, or start the worker manually instead."
    }
} else {
    Write-Host "skipped Scheduled Task registration. Start the worker manually with: $RunBat"
}

Write-Host ""
Write-Host "=== done ===" -ForegroundColor Green
Write-Host "test it now with: $RunBat --once"
