# CAMSPECTOR: one-command setup + GPU benchmark for a Windows machine with an NVIDIA GPU.
# Usage (PowerShell, from the repo folder):
#   powershell -ExecutionPolicy Bypass -File .\setup_and_bench.ps1
# Optional:  -Python "C:\path\to\python3.11.exe"   -Cuda cu126
#            -SkipInstall [-Python <existing env python>]   (re-run without reinstalling)
# Result: results_<COMPUTERNAME>.zip in this folder. Send that file back.
param(
    [string]$Python = "",
    [string]$Cuda = "cu126",
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Venv = Join-Path $Root ".venv"
$LogDir = Join-Path $Root "experiments"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Log = Join-Path $LogDir "setup_log.txt"
"=== setup_and_bench $(Get-Date -Format s) ===" | Out-File $Log -Encoding utf8

function Step($msg) { Write-Host "`n>>> $msg" -ForegroundColor Cyan; ">>> $msg" | Out-File $Log -Append -Encoding utf8 }

# Native tools log to stderr; in PowerShell 5.1 that must not count as an error,
# so success is judged by exit code only ($LASTEXITCODE after the call).
function Exec($exe, [string[]]$argv, [string]$tee = $Log) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $exe @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $tee -Append }
    finally { $ErrorActionPreference = $old }
}
function Run($exe, [string[]]$argv, [string]$tee = $Log) {
    Exec $exe $argv $tee
    if ($LASTEXITCODE -ne 0) { throw "command failed ($LASTEXITCODE): $exe $($argv -join ' ')" }
}

# Keep caches next to the repo, not on C:
$Cache = Join-Path $Root ".cache"
$env:PIP_CACHE_DIR = "$Cache\pip"; $env:HF_HOME = "$Cache\hf"; $env:TORCH_HOME = "$Cache\torch"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"
$env:PYTHONPATH = Join-Path $Root "code"

Step "Hardware"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { Exec "nvidia-smi" @() }
else { Write-Warning "nvidia-smi not found: is the NVIDIA driver installed?" }

if (-not $SkipInstall) {
    Step "Find Python 3.11"
    if (-not $Python -and (Get-Command py -ErrorAction SilentlyContinue)) {
        foreach ($v in "3.11", "3.12", "3.13") {
            $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            $ErrorActionPreference = $old
            if ($LASTEXITCODE -eq 0 -and $exe) { $Python = "$exe".Trim(); break }
        }
    }
    if (-not $Python -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $exe = & python -c "import sys; print(sys.executable if sys.version_info[:2] in [(3,11),(3,12),(3,13)] else '')"
        if ($exe) { $Python = "$exe".Trim() }
    }
    if (-not $Python) { throw "Python 3.11 (or 3.12/3.13) not found. Install 3.11 from python.org (tick 'Add to PATH') or pass -Python <path>." }
    $ver = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($ver -ne "3.11") { Write-Warning "Using Python $ver (project target is 3.11)." }
    Write-Host "Using $Python ($ver)"

    Step "Create venv"
    if (-not (Test-Path "$Venv\Scripts\python.exe")) { Run $Python @("-m", "venv", $Venv) }
    $Py = "$Venv\Scripts\python.exe"
    Run $Py @("-m", "pip", "install", "--upgrade", "pip")

    Step "Install PyTorch ($Cuda) -- ~3 GB download"
    Run $Py @("-m", "pip", "install", "torch==2.14.1", "torchvision==0.29.1", "--index-url", "https://download.pytorch.org/whl/$Cuda")

    Step "Install requirements"
    Run $Py @("-m", "pip", "install", "-r", "requirements.txt")
} else {
    # -SkipInstall: reuse the venv, or an existing interpreter given with -Python
    $Py = if ($Python) { $Python } else { "$Venv\Scripts\python.exe" }
}
Exec $Py @("-m", "pip", "freeze") "$LogDir\pip_freeze.txt" | Out-Null

Step "GPU check (CUDA + fp16)"
Exec $Py @("code\scripts\gpu_check.py", "--out", "experiments\gpu_check.json")
$gpuOk = ($LASTEXITCODE -eq 0)

Step "Generate synthetic test videos"
Run $Py @("code\scripts\make_synthetic_videos.py", "data\synthetic")

Step "Unit tests"
Exec $Py @("-m", "pytest", "code\tests", "-q", "-p", "no:cacheprovider") "$LogDir\pytest.txt"
$testsOk = ($LASTEXITCODE -eq 0)

Step "Encoder benchmark (GPU fp16 + CPU fp32)"
Run $Py @("code\scripts\bench_encoders.py", "--cpu", "--out", "experiments\bench_encoders_$($env:COMPUTERNAME).json") "$LogDir\bench_table_$($env:COMPUTERNAME).md"

Step "Pack results"
$zip = Join-Path $Root "results_$($env:COMPUTERNAME).zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$LogDir\*" -DestinationPath $zip
Write-Host "`nGPU check ok: $gpuOk   tests ok: $testsOk"
Write-Host "DONE. Send this file back: $zip" -ForegroundColor Green
