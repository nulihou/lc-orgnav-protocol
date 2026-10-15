# P15 runner. Verifies frozen hashes, waits for sibling experiments/P14/P14_DONE.txt,
# trains M1 -> M12 -> FULL on both tracks (resumable; up to 2 automatic resumes per step),
# predicts (P15_FINAL=1, hashed), and writes P15_DONE.txt only on success.
# Scoring is run separately afterwards.
Set-Location $PSScriptRoot
$py = if ($env:LCORGNAV_PYTHON) { $env:LCORGNAV_PYTHON } else { 'python' }
$codeRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\code')).Path
$env:PYTHONPATH = "$codeRoot;$PSScriptRoot"
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$P14 = (Resolve-Path (Join-Path $PSScriptRoot '..\P14')).Path
function L($m){ "$m $(Get-Date -f s)" | Add-Content p15.log }
function Dev($m){ "- $(Get-Date -f s) AUTO: $m" | Add-Content DEVIATION_runtime_auto.md }
function Fail($m){ L "FAIL $m"; exit 1 }
$man = Get-Content P15_FREEZE_MANIFEST.json -Raw | ConvertFrom-Json
foreach ($p in $man.files.PSObject.Properties) {
  $f = $p.Name
  if (-not [IO.Path]::IsPathRooted($f)) { $f = Join-Path $PSScriptRoot $f }
  if ((Get-FileHash $f -Algorithm SHA256).Hash.ToLower() -ne $p.Value) { Fail "hash mismatch $f" }
}
L 'START (hashes verified); waiting for P14'
$n = 0
while (-not (Test-Path (Join-Path $P14 'P14_DONE.txt'))) {
  if ($n % 30 -eq 0) { L 'WAIT P14 (no P14_DONE.txt yet; set P15_SKIP_P14_WAIT=1 to continue)' }
  if ($env:P15_SKIP_P14_WAIT -eq '1') {
    L 'P15_SKIP_P14_WAIT=1; starting without P14_DONE.txt'
    Dev 'Started P15 without P14_DONE.txt (P15_SKIP_P14_WAIT=1)'
    break
  }
  $n++; Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 5
L 'P14 ready or wait skipped; training starts'
function Step($name, $argsList, $log) {
  for ($try = 0; $try -le 2; $try++) {
    if ($try -gt 0) {
      L "RETRY $name attempt $try (resume from checkpoints)"
      Dev "$name failed; automatic resume attempt $try from last.pt"
    }
    L "START $name"
    & $py @argsList 1>> "$log.log" 2>> "$log.err"
    if ($LASTEXITCODE -eq 0) { L "EXIT $name 0"; return }
    L "ERROR $name exit $LASTEXITCODE"
    Start-Sleep -Seconds 20
  }
  Fail "$name after retries"
}
foreach ($a in 'M1','M12','FULL') {
  foreach ($t in 'hard_confusable','scale300') {
    Step "train $a $t" @('p15_train.py', $a, $t) "train_${a}_${t}"
  }
}
$env:P15_FINAL = '1'
Step 'predict' @('p15_predict.py') 'predict'
if (-not (Test-Path PREDICTION_HASHES_P15.json)) { Fail 'no prediction hashes' }
"done $(Get-Date -f s)" | Set-Content P15_DONE.txt
L 'DONE'