param(
    [switch]$Rebuild,
    [int]$Samples = 32
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Blender = 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe'
$Python = Join-Path (Split-Path -Parent $Root) '.venv\Scripts\python.exe'
$Scenes = @('car-turn', 'drift-straight', 'soccerball')
$FFmpeg = Get-ChildItem "$Root\tools\ffmpeg" -Filter ffmpeg.exe -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName

if (-not (Test-Path -LiteralPath $Blender)) {
    throw "Blender not found at $Blender"
}
if (-not $FFmpeg) {
    throw "Portable FFmpeg not found below $Root\tools\ffmpeg"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Workspace Python not found at $Python"
}

foreach ($Name in $Scenes) {
    $Blend = Join-Path $Root "scenes\$Name.blend"
    if ($Rebuild -or -not (Test-Path -LiteralPath $Blend)) {
        & $Blender --background --factory-startup --python "$Root\scripts\generate_scenes.py" -- --scene $Name --samples $Samples
        if ($LASTEXITCODE -ne 0) { throw "Scene build failed: $Name" }
    }
    & $Blender --background $Blend --render-anim
    if ($LASTEXITCODE -ne 0) { throw "Frame render failed: $Name" }
    & $FFmpeg -y -hide_banner -loglevel warning -framerate 24 -start_number 1 `
        -i "$Root\renders\$Name\frame_%04d.png" -c:v libx264 -preset slow -crf 16 `
        -pix_fmt yuv420p -movflags +faststart -r 24 "$Root\videos\${Name}_anomaly.mp4"
    if ($LASTEXITCODE -ne 0) { throw "Video encode failed: $Name" }
    & $Blender --background --factory-startup --python "$Root\scripts\make_contact_sheet.py" -- --scene $Name
    if ($LASTEXITCODE -ne 0) { throw "Contact sheet failed: $Name" }
}

& $Python "$Root\scripts\verify_outputs.py"
if ($LASTEXITCODE -ne 0) { throw "Final verification failed" }
Write-Host "All Blender anomaly videos completed under $Root"
