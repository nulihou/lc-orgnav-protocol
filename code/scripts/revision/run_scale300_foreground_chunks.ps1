param(
  [string[]]$Methods = @("random", "mlp_pooling", "ours", "rule_semantic", "oracle"),
  [int]$RunsPerScene = 5,
  [int]$ChunkStart = 0,
  [int]$ChunkEnd = 299,
  [string]$Device = "cpu",
  [bool]$Resume = $true,
  [switch]$StartCarla,
  [int]$CarlaWaitSeconds = 45,
  [string]$RunTag = "",
  [string]$OutputRoot = "",
  [string]$LogRoot = "",
  [string]$ClosedLoopRoot = "",
  [string]$RunRootOverride = "",
  [string]$ScenarioChunkDir = "",
  [string]$ProtocolDirOverride = "",
  [string]$SeedConfigOverride = "",
  [string]$MlpCheckpointOverride = "",
  [string]$OursCheckpointOverride = "",
  [string]$ControllerProfile = "improved_v4",
  [int]$MaxSteps = 1000,
  [int]$CarlaPort = 2000,
  [int]$TmPort = 8000,
  [ValidateSet("summary", "all", "silent")]
  [string]$ConsoleMode = "summary",
  [switch]$MergeWhenComplete
)

$ErrorActionPreference = "Stop"

if ($RunTag -eq "") {
  $RunTag = Get-Date -Format "yyyyMMdd_HHmmss"
}

$Methods = @($Methods | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
if ($RunRootOverride -ne "") {
  $RunRoot = (Resolve-Path -LiteralPath $RunRootOverride).Path
} else {
  $RunRoot = Join-Path $Root "runs\scale300_20260522"
}
if ($ScenarioChunkDir -ne "") {
  $ChunkDir = (Resolve-Path -LiteralPath $ScenarioChunkDir).Path
} else {
  $ChunkDir = Join-Path $RunRoot "scenarios\chunks"
}
if ($ProtocolDirOverride -ne "") {
  $ProtocolDir = (Resolve-Path -LiteralPath $ProtocolDirOverride).Path
} else {
  $ProtocolDir = Join-Path $RunRoot "protocol_graphs\test"
}
if ($SeedConfigOverride -ne "") {
  $SeedConfig = (Resolve-Path -LiteralPath $SeedConfigOverride).Path
} else {
  $SeedConfig = Join-Path $RunRoot "scenarios\test_seed_config_300x5.json"
}
$MlpCheckpoint = if ($MlpCheckpointOverride -ne "") {
  (Resolve-Path -LiteralPath $MlpCheckpointOverride).Path
} else {
  "runs\scale300_20260522\training\candidate_mlp_seed7\candidate_mlp_best.pt"
}
$OursCheckpoint = if ($OursCheckpointOverride -ne "") {
  (Resolve-Path -LiteralPath $OursCheckpointOverride).Path
} else {
  "runs\scale300_20260522\training\lcorg_full_relgate_goalrank_seed21\full_best.pt"
}

if ($OutputRoot -eq "") {
  $OutputRoot = Join-Path $RunRoot ("closed_loop_chunks_foreground\" + $RunTag)
}
if ($LogRoot -eq "") {
  $LogRoot = Join-Path $RunRoot ("closed_loop_foreground_logs\" + $RunTag)
}
if ($ClosedLoopRoot -eq "") {
  $ClosedLoopRoot = Join-Path $RunRoot ("closed_loop_foreground\" + $RunTag)
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$Activate = if ($env:CONDA_ACTIVATE_BAT) { $env:CONDA_ACTIVATE_BAT } else { throw "Set CONDA_ACTIVATE_BAT" }
$EnvPath = if ($env:LCORGNAV_CARLA_ENV) { $env:LCORGNAV_CARLA_ENV } else { throw "Set LCORGNAV_CARLA_ENV" }
$CarlaExe = if ($env:CARLA_UE4_EXE) { $env:CARLA_UE4_EXE } else { throw "Set CARLA_UE4_EXE" }
$CarlaApi = if ($env:CARLA_PYTHON_API_PATH) { $env:CARLA_PYTHON_API_PATH } else { throw "Set CARLA_PYTHON_API_PATH" }
$CarlaEgg = if ($env:CARLA_EGG) { $env:CARLA_EGG } else { throw "Set CARLA_EGG" }

function Get-JsonlLineCount([string]$Path) {
  if (!(Test-Path -LiteralPath $Path)) { return 0 }
  return (Get-Content -LiteralPath $Path | Where-Object { $_.Trim() -ne "" } | Measure-Object -Line).Lines
}

function Get-ChunkSceneCount([string]$Path) {
  $json = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
  return @($json).Count
}

function Read-ChunkSceneIds([string]$Path) {
  $json = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
  return @($json | ForEach-Object { [int]$_.id })
}

function Add-LogLine([string]$LogPath, [string]$Line) {
  for ($attempt = 1; $attempt -le 5; $attempt++) {
    try {
      Add-Content -LiteralPath $LogPath -Value $Line -Encoding utf8 -ErrorAction Stop
      return
    } catch {
      if ($attempt -eq 5) { throw }
      Start-Sleep -Milliseconds (200 * $attempt)
    }
  }
}

function Invoke-ForegroundLogged([string]$Command, [string]$LogPath) {
  $full = "call `"$Activate`" `"$EnvPath`" && set CARLA_ROOT=%CARLA_ROOT% && set CARLA_PYTHON_API_PATH=$CarlaApi && set PYTHONPATH=$CarlaApi;$CarlaEgg;%PYTHONPATH% && cd /d `"$Root`" && $Command"
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Set-Content -LiteralPath $LogPath -Value "[$stamp] CMD: $Command" -Encoding utf8
  $oldErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
  cmd.exe /c $full 2>&1 | ForEach-Object {
    $line = [string]$_
    if ($ConsoleMode -eq "all") {
      Write-Host $line
    } elseif ($ConsoleMode -eq "summary") {
      if (
        $line -match "^\[Config\] Scenarios:" -or
        $line -match "^\[Config\] Output:" -or
        $line -match "^\[Config\] Runs per scene:" -or
        $line -match "^\[Batch\]" -or
        $line -match "^\[Done\]" -or
        $line -match "Wrote .*episodes" -or
        $line -match "Traceback|PermissionError|WinError|Access is denied|鎷掔粷璁块棶|璇勪及鍑洪敊|ERROR"
      ) {
        Write-Host $line
      }
    }
    Add-LogLine -LogPath $LogPath -Line $line
  }
  } finally {
    $ErrorActionPreference = $oldErrorActionPreference
  }
  $exitCode = $LASTEXITCODE
  Add-LogLine -LogPath $LogPath -Line "[exit_code] $exitCode"
  if ($exitCode -ne 0) {
    throw "Command failed with exit code $exitCode. See $LogPath"
  }
}

function Test-ChunkJsonl(
  [string]$Method,
  [string]$ChunkLabel,
  [string]$JsonlPath,
  [int[]]$ExpectedSceneIds,
  [int]$RunsPerScene,
  [string]$LogPath
) {
  if (!(Test-Path -LiteralPath $JsonlPath)) {
    throw "Missing JSONL for ${Method} ${ChunkLabel}: $JsonlPath"
  }

  $lines = @(Get-Content -LiteralPath $JsonlPath | Where-Object { $_.Trim() -ne "" })
  $expectedRows = $ExpectedSceneIds.Count * $RunsPerScene
  if ($lines.Count -ne $expectedRows) {
    throw "Wrong row count for ${Method} ${ChunkLabel}: $($lines.Count)/$expectedRows"
  }

  $seen = @{}
  $sceneCounts = @{}
  $validRows = 0
  $success8 = 0
  $success4 = 0
  $missingFields = New-Object System.Collections.Generic.List[string]
  $duplicateKeys = New-Object System.Collections.Generic.List[string]
  $required = @("run_id", "valid", "success_8m", "success_4m", "collision", "lane_invasion", "min_ttc_s", "min_distance_to_vehicle_m")

  for ($i = 0; $i -lt $lines.Count; $i++) {
    try {
      $row = $lines[$i] | ConvertFrom-Json
    } catch {
      throw "Invalid JSON at ${JsonlPath}:$($i + 1): $($_.Exception.Message)"
    }
    $sid = $null
    if ($null -ne $row.scene_id) { $sid = [int]$row.scene_id }
    elseif ($null -ne $row.scen_id) { $sid = [int]$row.scen_id }
    else { $missingFields.Add("line $($i + 1): scene_id/scen_id") | Out-Null }

    foreach ($field in $required) {
      if ($row.PSObject.Properties.Name -notcontains $field) {
        $missingFields.Add("line $($i + 1): $field") | Out-Null
      }
    }

    $rid = if ($null -ne $row.run_id) { [int]$row.run_id } else { -1 }
    $key = "$sid/$rid"
    if ($seen.ContainsKey($key)) {
      $duplicateKeys.Add($key) | Out-Null
    } else {
      $seen[$key] = $true
    }
    if ($null -ne $sid) {
      if (!$sceneCounts.ContainsKey($sid)) { $sceneCounts[$sid] = 0 }
      $sceneCounts[$sid] += 1
    }
    if ($row.valid -ne $false) { $validRows += 1 }
    if ($row.success_8m -eq $true) { $success8 += 1 }
    if ($row.success_4m -eq $true) { $success4 += 1 }
  }

  if ($duplicateKeys.Count -gt 0) {
    throw "Duplicate scene/run keys in ${Method} ${ChunkLabel}: $($duplicateKeys -join ', ')"
  }
  if ($missingFields.Count -gt 0) {
    throw "Missing required fields in ${Method} ${ChunkLabel}: $($missingFields[0..([Math]::Min($missingFields.Count - 1, 20))] -join '; ')"
  }
  foreach ($sid in $ExpectedSceneIds) {
    if (!$sceneCounts.ContainsKey($sid) -or [int]$sceneCounts[$sid] -ne $RunsPerScene) {
      throw "Scene $sid has $($sceneCounts[$sid]) rows in ${Method} ${ChunkLabel}; expected $RunsPerScene"
    }
  }

  $logRaw = ""
  if (Test-Path -LiteralPath $LogPath) {
    $logRaw = Get-Content -LiteralPath $LogPath -Raw
  }
  $fatalPatterns = @("Traceback", "PermissionError", "WinError", "Access is denied", "鎷掔粷璁块棶", "璇勪及鍑洪敊", "Command failed")
  $fatalHits = @()
  foreach ($pattern in $fatalPatterns) {
    if ($logRaw -match [regex]::Escape($pattern)) {
      $fatalHits += $pattern
    }
  }
  if ($fatalHits.Count -gt 0) {
    throw "Fatal log pattern(s) in ${Method} ${ChunkLabel}: $($fatalHits -join ', '). See $LogPath"
  }

  $audit = [ordered]@{
    method = $Method
    chunk = $ChunkLabel
    jsonl = $JsonlPath
    log = $LogPath
    expected_rows = $expectedRows
    actual_rows = $lines.Count
    expected_scenes = $ExpectedSceneIds.Count
    valid_rows = $validRows
    success_8m = $success8
    success_4m = $success4
    checked_at = (Get-Date).ToString("s")
  }
  $auditPath = [System.IO.Path]::ChangeExtension($JsonlPath, ".audit.json")
  $audit | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $auditPath -Encoding utf8
  Write-Host "[AUDIT] $Method $ChunkLabel rows=$($lines.Count)/$expectedRows valid=$validRows success8=$success8 success4=$success4"
}

if ($StartCarla) {
  $existing = Get-Process | Where-Object { $_.ProcessName -match "Carla|UE4" }
  if (-not $existing) {
    Write-Host "[CARLA] Starting server..."
    Start-Process -FilePath $CarlaExe -ArgumentList @("-carla-server", "-quality-level=Low", "-ResX=640", "-ResY=480", "-RenderOffScreen") -WorkingDirectory (Split-Path -Parent $CarlaExe) -WindowStyle Hidden
    Start-Sleep -Seconds $CarlaWaitSeconds
  } else {
    Write-Host "[CARLA] Existing CARLA/UE4 process detected."
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
    checkpoint = $MlpCheckpoint
  }
  ours = @{
    script = "scripts\online_multi\eval_gs_proto_driven.py"
    dir = "ours"
    prefix = "ours"
    extra = "--device $Device"
    checkpoint = $OursCheckpoint
  }
  rule_semantic = @{
    script = "scripts\online_multi\eval_gs_proto_driven.py"
    dir = "rule_semantic"
    prefix = "rule_semantic"
    extra = "--model-type rule_semantic --selector-profile rule_semantic --device $Device"
    checkpoint = $null
  }
  fine_rank_rule = @{
    script = "scripts\online_multi\eval_gs_proto_driven.py"
    dir = "fine_rank_rule"
    prefix = "fine_rank_rule"
    extra = "--model-type rule_semantic --selector-profile fine_rank_rule --device $Device"
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

$runManifest = [ordered]@{
  run_tag = $RunTag
  output_root = $OutputRoot
  log_root = $LogRoot
  closed_loop_root = $ClosedLoopRoot
  run_root = $RunRoot
  chunk_dir = $ChunkDir
  protocol_dir = $ProtocolDir
  seed_config = $SeedConfig
  mlp_checkpoint = $MlpCheckpoint
  ours_checkpoint = $OursCheckpoint
  methods = $Methods
  chunk_start = $ChunkStart
  chunk_end = $ChunkEnd
  runs_per_scene = $RunsPerScene
  controller_profile = $ControllerProfile
  max_steps = $MaxSteps
  carla_port = $CarlaPort
  tm_port = $TmPort
  started_at = (Get-Date).ToString("s")
}
$runManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $LogRoot "run_manifest.json") -Encoding utf8

foreach ($method in $Methods) {
  if (!$methodConfigs.ContainsKey($method)) { throw "Unknown method: $method" }
  $cfg = $methodConfigs[$method]
  $outDir = Join-Path $OutputRoot $cfg.dir
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
    $sceneIds = @(Read-ChunkSceneIds $chunk.FullName)
    $expected = $sceneIds.Count * $RunsPerScene
    $out = Join-Path $outDir ("{0}_{1}_{2}.jsonl" -f $cfg.prefix, $lo, $hi)
    $log = Join-Path $LogRoot ("{0}_{1}_{2}.log" -f $cfg.prefix, $lo, $hi)
    $existingLines = Get-JsonlLineCount $out
    if ($Resume -and $existingLines -eq $expected) {
      Write-Host "[CHECK] $method $lo-$hi existing rows=$existingLines/$expected"
      Test-ChunkJsonl -Method $method -ChunkLabel "$lo-$hi" -JsonlPath $out -ExpectedSceneIds $sceneIds -RunsPerScene $RunsPerScene -LogPath $log
      Write-Host "[SKIP] $method $lo-$hi complete"
      continue
    }
    if (Test-Path -LiteralPath $out) {
      $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
      $stale = "$out.stale.$stamp"
      Move-Item -LiteralPath $out -Destination $stale -Force
      Write-Host "[STALE] moved incomplete output to $stale"
      if (Test-Path -LiteralPath $log) {
        $staleLog = "$log.stale.$stamp"
        Move-Item -LiteralPath $log -Destination $staleLog -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $log) {
          $log = "$log.retry.$stamp"
        }
        Write-Host "[STALE] isolated incomplete log to $staleLog"
      }
    }

    $cmd = "python $($cfg.script) --scenarios `"$($chunk.FullName)`" --protocol-dir `"$ProtocolDir`" --output `"$out`" --seed-config `"$SeedConfig`" --num-runs-per-scene $RunsPerScene --max-scenes 0 --max-steps $MaxSteps --controller-profile $ControllerProfile --no-rendering --carla-port $CarlaPort --tm-port $TmPort"
    if ($cfg.checkpoint) {
      $cmd += " --checkpoint `"$($cfg.checkpoint)`""
    }
    if ($cfg.extra) {
      $cmd += " $($cfg.extra)"
    }
    Write-Host "[RUN] $method $lo-$hi expected_rows=$expected -> $out"
    Invoke-ForegroundLogged $cmd $log
    Test-ChunkJsonl -Method $method -ChunkLabel "$lo-$hi" -JsonlPath $out -ExpectedSceneIds $sceneIds -RunsPerScene $RunsPerScene -LogPath $log
  }
}

if ($MergeWhenComplete) {
  Write-Host "[MERGE] Combining foreground chunks into $ClosedLoopRoot"
  New-Item -ItemType Directory -Path $ClosedLoopRoot -Force | Out-Null
  Invoke-ForegroundLogged "python scripts\revision\merge_scale300_jsonl.py --chunks-root `"$OutputRoot`" --output-root `"$ClosedLoopRoot`"" (Join-Path $LogRoot "merge.log")
  Write-Host "[MERGE] Complete. Summarize with: python scripts\revision\summarize_scale300_closed_loop.py --run-root runs\scale300_20260522 --closed-root `"$ClosedLoopRoot`""
}

Write-Host "[DONE] Foreground scale-300 chunks completed for RunTag=$RunTag"
