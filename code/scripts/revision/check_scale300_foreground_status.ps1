param(
  [string]$RunTag = "scale300_fg_full_clean_20260523",
  [string]$RunRoot = "runs\scale300_20260522",
  [string[]]$Methods = @("random", "mlp_pooling", "ours", "rule_semantic", "oracle"),
  [int]$RunsPerScene = 5,
  [int]$ChunkStart = 0,
  [int]$ChunkEnd = 299,
  [switch]$TailLatestLog,
  [int]$TailLines = 30
)

$ErrorActionPreference = "Stop"

$chunkRoot = Join-Path $RunRoot ("closed_loop_chunks_foreground\" + $RunTag)
$logRoot = Join-Path $RunRoot ("closed_loop_foreground_logs\" + $RunTag)
$statusPath = Join-Path $RunRoot ("FOREGROUND_PROGRESS_" + $RunTag + ".md")
$Methods = @($Methods | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })

function Get-ExpectedRowsForChunk([string]$BaseName) {
  if ($BaseName -match ".*_(\d{3})_(\d{3})$") {
    $lo = [int]$Matches[1]
    $hi = [int]$Matches[2]
    return (($hi - $lo + 1) * $RunsPerScene)
  }
  return (30 * $RunsPerScene)
}

function Get-ExpectedChunkCount() {
  $chunkDir = Join-Path $RunRoot "scenarios\chunks"
  if (!(Test-Path -LiteralPath $chunkDir)) { return 0 }
  $chunks = @(Get-ChildItem -LiteralPath $chunkDir -Filter "test_scenarios_*.json" -File | Where-Object {
    if ($_.BaseName -match "test_scenarios_(\d{3})_(\d{3})") {
      ([int]$Matches[1] -ge $ChunkStart) -and ([int]$Matches[2] -le $ChunkEnd)
    } else {
      $false
    }
  })
  return $chunks.Count
}

$expectedChunkCount = Get-ExpectedChunkCount

function Count-Lines([string]$Path) {
  if (!(Test-Path -LiteralPath $Path)) { return 0 }
  return (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

function Method-Status([string]$Method) {
  $dir = Join-Path $chunkRoot $Method
  $files = @()
  if (Test-Path -LiteralPath $dir) {
    $files = @(Get-ChildItem -LiteralPath $dir -Filter "*.jsonl" -File | Sort-Object Name)
  }
  $complete = 0
  $rows = 0
  $expectedRowsTotal = 0
  $incomplete = @()
  foreach ($f in $files) {
    $n = Count-Lines $f.FullName
    $expectedRows = Get-ExpectedRowsForChunk $f.BaseName
    $rows += $n
    $expectedRowsTotal += $expectedRows
    if ($n -eq $expectedRows) {
      $complete += 1
    } else {
      $incomplete += ("{0}:{1}/{2}" -f $f.BaseName, $n, $expectedRows)
    }
  }
  if ($expectedRowsTotal -eq 0) {
    $expectedRowsTotal = $expectedChunkCount * 30 * $RunsPerScene
  }
  [pscustomobject]@{
    Method = $Method
    Chunks = ("{0}/{1}" -f $complete, $expectedChunkCount)
    Rows = ("{0}/{1}" -f $rows, $expectedRowsTotal)
    Incomplete = if ($incomplete.Count) { $incomplete -join "; " } else { "" }
  }
}

$summary = @($Methods | ForEach-Object { Method-Status $_ })
$active = @(Get-Process | Where-Object { $_.ProcessName -match "Carla|UE4|python|cmd|powershell" } |
  Select-Object Id, ProcessName, CPU, StartTime, Path |
  Sort-Object ProcessName, Id)

$gpu = ""
try {
  $gpu = (& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null) -join "`n"
} catch {
  $gpu = "nvidia-smi unavailable"
}

$latestLog = $null
if (Test-Path -LiteralPath $logRoot) {
  $latestLog = Get-ChildItem -LiteralPath $logRoot -File -Filter "*.log" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
}

$now = Get-Date
Write-Host "Scale-300 foreground status ($RunTag) @ $now"
Write-Host ""
$summary | Format-Table -AutoSize
Write-Host ""
Write-Host "Active CARLA/Python-related processes:"
$active | Format-Table Id,ProcessName,CPU,StartTime -AutoSize
Write-Host ""
Write-Host "GPU:"
Write-Host $gpu
if ($latestLog) {
  Write-Host ""
  Write-Host ("Latest log: {0} ({1})" -f $latestLog.FullName, $latestLog.LastWriteTime)
  if ($TailLatestLog) {
    Write-Host ""
    Get-Content -LiteralPath $latestLog.FullName -Tail $TailLines
  }
}

$lines = @()
$lines += "# Scale-300 Foreground Progress"
$lines += ""
$lines += ("- RunTag: {0}" -f $RunTag)
$lines += "- Updated: $now"
$lines += ("- Chunk root: {0}" -f $chunkRoot)
$lines += ("- Log root: {0}" -f $logRoot)
$lines += ""
$lines += "## Chunk Progress"
$lines += ""
$lines += "| Method | Chunks | Rows | Incomplete |"
$lines += "|---|---:|---:|---|"
foreach ($row in $summary) {
  $lines += "| $($row.Method) | $($row.Chunks) | $($row.Rows) | $($row.Incomplete) |"
}
$lines += ""
$lines += "## Active Processes"
$lines += ""
$lines += "| PID | Process | CPU seconds | Start time |"
$lines += "|---:|---|---:|---|"
foreach ($p in $active) {
  $lines += "| $($p.Id) | $($p.ProcessName) | $([math]::Round([double]$p.CPU, 1)) | $($p.StartTime) |"
}
$lines += ""
$lines += "## GPU"
$lines += ""
$lines += "```text"
$lines += $gpu
$lines += '```'
if ($latestLog) {
  $lines += ""
  $lines += ("- Latest log: {0}" -f $latestLog.FullName)
}
$lines | Set-Content -LiteralPath $statusPath -Encoding utf8
Write-Host ""
Write-Host "Wrote status file: $statusPath"
