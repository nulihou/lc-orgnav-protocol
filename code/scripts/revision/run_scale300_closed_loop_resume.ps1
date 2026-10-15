param(
  [string[]]$Methods = @("random", "mlp_pooling", "ours", "rule_semantic", "oracle"),
  [int]$RunsPerScene = 5,
  [int]$ChunkStart = 0,
  [int]$ChunkEnd = 299,
  [string]$Device = "cpu",
  [switch]$StartCarla,
  [switch]$SkipCompleted = $true,
  [int]$CarlaWaitSeconds = 45,
  [string]$ControllerProfile = "improved_v4",
  [int]$MaxSteps = 1000,
  [string]$RunTag = ""
)

$ErrorActionPreference = "Stop"

$Methods = @($Methods | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$RunRoot = Join-Path $Root "runs\scale300_20260522"
$ChunkDir = Join-Path $RunRoot "scenarios\chunks"
$ProtocolDir = Join-Path $RunRoot "protocol_graphs\test"
$SeedConfig = Join-Path $RunRoot "scenarios\test_seed_config_300x5.json"
$LogDir = Join-Path $RunRoot "closed_loop_run_logs"
if ($RunTag -ne "") {
  $LogDir = Join-Path $LogDir $RunTag
}
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$Activate = if ($env:CONDA_ACTIVATE_BAT) { $env:CONDA_ACTIVATE_BAT } else { throw "Set CONDA_ACTIVATE_BAT" }
$EnvPath = if ($env:LCORGNAV_CARLA_ENV) { $env:LCORGNAV_CARLA_ENV } else { throw "Set LCORGNAV_CARLA_ENV" }
$CarlaExe = if ($env:CARLA_UE4_EXE) { $env:CARLA_UE4_EXE } else { throw "Set CARLA_UE4_EXE" }
$CarlaApi = if ($env:CARLA_PYTHON_API_PATH) { $env:CARLA_PYTHON_API_PATH } else { throw "Set CARLA_PYTHON_API_PATH" }
$CarlaEgg = if ($env:CARLA_EGG) { $env:CARLA_EGG } else { throw "Set CARLA_EGG" }

function Get-JsonlLineCount([string]$Path) {
  if (!(Test-Path -LiteralPath $Path)) { return 0 }
  return (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

function Get-ChunkSceneCount([string]$Path) {
  $json = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
  return @($json).Count
}

function Invoke-CmdLogged([string]$Command, [string]$LogPath) {
  $full = "call `"$Activate`" `"$EnvPath`" && set CARLA_ROOT=%CARLA_ROOT% && set CARLA_PYTHON_API_PATH=$CarlaApi && set PYTHONPATH=$CarlaApi;$CarlaEgg;%PYTHONPATH% && cd /d `"$Root`" && $Command"
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Add-Content -LiteralPath $LogPath -Value "[$stamp] CMD: $Command"
  cmd.exe /c $full 2>&1 | Tee-Object -FilePath $LogPath -Append
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE. See $LogPath"
  }
}

if ($StartCarla) {
  $existing = Get-Process | Where-Object { $_.ProcessName -match "Carla|UE4" }
  if (-not $existing) {
    Start-Process -FilePath $CarlaExe -ArgumentList @("-carla-server", "-quality-level=Low", "-ResX=640", "-ResY=480", "-RenderOffScreen") -WorkingDirectory (Split-Path -Parent $CarlaExe) -WindowStyle Hidden
    Start-Sleep -Seconds $CarlaWaitSeconds
  }
}

$chunks = Get-ChildItem -LiteralPath $ChunkDir -Filter "test_scenarios_*.json" | Sort-Object Name
if (!$chunks) { throw "No test_scenarios_*.json chunks found in $ChunkDir" }

$methodConfigs = @{
  random = @{
    script = "scripts\online_multi\eval_random_proto_driven.py"
    dir = "random"
    prefix = "random"
    extra = ""
    checkpoint = $null
  }
  mlp_pooling = @{
    script = "scripts\online_multi\eval_gs_proto_driven.py"
    dir = "mlp_pooling"
    prefix = "mlp_pooling"
    extra = "--model-type candidate_mlp --device $Device"
    checkpoint = "runs\scale300_20260522\training\candidate_mlp_seed7\candidate_mlp_best.pt"
  }
  ours = @{
    script = "scripts\online_multi\eval_gs_proto_driven.py"
    dir = "ours"
    prefix = "ours"
    extra = "--device $Device"
    checkpoint = "runs\scale300_20260522\training\lcorg_full_relgate_goalrank_seed21\full_best.pt"
  }
  rule_semantic = @{
    script = "scripts\online_multi\eval_gs_proto_driven.py"
    dir = "rule_semantic"
    prefix = "rule_semantic"
    extra = "--model-type rule_semantic --selector-profile rule_semantic --device $Device"
    checkpoint = $null
  }
  oracle = @{
    script = "scripts\online_multi\eval_oracle_proto_driven.py"
    dir = "oracle"
    prefix = "oracle"
    extra = ""
    checkpoint = $null
  }
}

foreach ($method in $Methods) {
  if (!$methodConfigs.ContainsKey($method)) { throw "Unknown method: $method" }
  $cfg = $methodConfigs[$method]
  $outDir = Join-Path $RunRoot ("closed_loop_chunks\" + $cfg.dir)
  New-Item -ItemType Directory -Path $outDir -Force | Out-Null
  foreach ($chunk in $chunks) {
    if ($chunk.BaseName -notmatch "test_scenarios_(\d{3})_(\d{3})") {
      throw "Unexpected chunk name: $($chunk.Name)"
    }
    $lo = $Matches[1]
    $hi = $Matches[2]
    if ([int]$lo -lt $ChunkStart -or [int]$hi -gt $ChunkEnd) {
      continue
    }
    $expected = (Get-ChunkSceneCount $chunk.FullName) * $RunsPerScene
    $out = Join-Path $outDir ("{0}_{1}_{2}.jsonl" -f $cfg.prefix, $lo, $hi)
    $existingLines = Get-JsonlLineCount $out
    if ($SkipCompleted -and $existingLines -ge $expected) {
      Write-Host "[SKIP] $method $lo-$hi already has $existingLines/$expected rows"
      continue
    }

    $log = Join-Path $LogDir ("{0}_{1}_{2}.log" -f $cfg.prefix, $lo, $hi)
    $cmd = "python $($cfg.script) --scenarios `"$($chunk.FullName)`" --protocol-dir `"$ProtocolDir`" --output `"$out`" --seed-config `"$SeedConfig`" --num-runs-per-scene $RunsPerScene --max-scenes 0 --max-steps $MaxSteps --controller-profile $ControllerProfile --no-rendering"
    if ($cfg.checkpoint) {
      $cmd += " --checkpoint `"$($cfg.checkpoint)`""
    }
    if ($cfg.extra) {
      $cmd += " $($cfg.extra)"
    }
    Write-Host "[RUN] $method $lo-$hi -> $out"
    Invoke-CmdLogged $cmd $log
    $newLines = Get-JsonlLineCount $out
    if ($newLines -lt $expected) {
      throw "Chunk finished with too few rows for ${method} ${lo}-${hi}: $newLines/$expected"
    }
  }
}

$allMethods = @("random", "mlp_pooling", "ours", "rule_semantic", "oracle")
$isFullRun = ($RunsPerScene -eq 5) -and ($ChunkStart -le 0) -and ($ChunkEnd -ge 299) -and (@($Methods | Sort-Object) -join ",") -eq (@($allMethods | Sort-Object) -join ",")
if ($isFullRun) {
  Write-Host "[MERGE] Combining chunks and summarizing"
  Invoke-CmdLogged "python scripts\revision\merge_scale300_jsonl.py --chunks-root runs\scale300_20260522\closed_loop_chunks --output-root runs\scale300_20260522\closed_loop" (Join-Path $LogDir "merge.log")
  Invoke-CmdLogged "python scripts\revision\summarize_scale300_closed_loop.py --run-root runs\scale300_20260522" (Join-Path $LogDir "summarize.log")
} else {
  Write-Host "[INFO] Partial/smoke run completed; skipping merge/summarize."
}
Write-Host "[DONE] Scale-300 closed-loop runner completed."
