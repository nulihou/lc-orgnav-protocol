param(
    [string]$Checkpoint = "runs\scale300_20260522\training\lcorg_full_relgate_goalrank_seed21\full_best.pt",
    [string]$Output = "runs\variant_consistency_20260528\main30_graph_relation_gate_30x10\graph_relation_gate_main30_30x10.jsonl",
    [int]$RunsPerScene = 10,
    [int]$MaxScenes = 30,
    [int]$MaxSteps = 1000,
    [string]$ControllerProfile = "improved_v4",
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path ".").Path
$outputPath = Join-Path $repo $Output
$outputDir = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = [System.IO.Path]::ChangeExtension($outputPath, ".$stamp.log")
$errPath = [System.IO.Path]::ChangeExtension($outputPath, ".$stamp.err.log")

$cmdLine = @(
    'call "%CONDA_ACTIVATE_BAT%" "%LCORGNAV_CARLA_ENV%"',
    'set CARLA_ROOT=%CARLA_ROOT%',
    'set CARLA_PYTHON_API_PATH=%CARLA_PYTHON_API_PATH%',
    'set PYTHONPATH=%CARLA_PYTHON_API_PATH%;%CARLA_EGG%;%PYTHONPATH%',
    "cd /d `"$repo`"",
    "python scripts\online_multi\eval_gs_proto_driven.py --scenarios data\online_multi\multi_seen_v3.json --protocol-dir scripts\online_multi\protocol_graphs --output `"$Output`" --seed-config runs\scaleup_20260520\main_30x10\random_seeds_30x10.json --num-runs-per-scene $RunsPerScene --max-scenes $MaxScenes --max-steps $MaxSteps --controller-profile $ControllerProfile --no-rendering --checkpoint `"$Checkpoint`" --device $Device --selector-profile relation_gated"
) -join " && "

Write-Host "[Run] Main-30 Graph+Relation-Gate"
Write-Host "[Run] Output: $outputPath"
Write-Host "[Run] Log: $logPath"

Push-Location $repo
try {
    & cmd.exe /c $cmdLine > $logPath 2> $errPath
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host "[Run] ExitCode: $exitCode"
if ($exitCode -ne 0) {
    if (Test-Path -LiteralPath $errPath) {
        Get-Content -LiteralPath $errPath -Tail 80
    }
    throw "Main-30 Graph+Relation-Gate run failed with exit code $exitCode."
}

if (Test-Path -LiteralPath $outputPath) {
    $count = (Get-Content -LiteralPath $outputPath | Measure-Object -Line).Lines
    Write-Host "[Run] Rows: $count"
} else {
    throw "Expected output was not created: $outputPath"
}
