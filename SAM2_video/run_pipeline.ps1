$ErrorActionPreference = "Stop"

$Base = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Base "runtime\env\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project-local Python environment is missing: $Python"
}

$env:TEMP = Join-Path $Base "runtime\tmp"
$env:TMP = $env:TEMP
$env:HOME = Join-Path $Base "runtime\home"
$env:HF_HOME = Join-Path $Base "runtime\cache\huggingface"
$env:TORCH_HOME = Join-Path $Base "runtime\cache\torch"
$env:XDG_CACHE_HOME = Join-Path $Base "runtime\cache"

New-Item -ItemType Directory -Force $env:TEMP | Out-Null
New-Item -ItemType Directory -Force $env:HOME | Out-Null
New-Item -ItemType Directory -Force $env:XDG_CACHE_HOME | Out-Null

& $Python (Join-Path $Base "scripts\prepare_sources.py")
& $Python (Join-Path $Base "scripts\run_sam2_tracking.py")
& $Python (Join-Path $Base "scripts\run_propainter.py")
& $Python (Join-Path $Base "scripts\build_counterfactuals.py")
& $Python (Join-Path $Base "scripts\audit_outputs.py")
& $Python (Join-Path $Base "scripts\finalize_manifest.py")

Write-Host "Completed. Deliverables are under $Base\outputs"
