Set-Location $PSScriptRoot
# Portable runner: uses `python` on PATH. Optional override: $env:LCORGNAV_PYTHON
$py = if ($env:LCORGNAV_PYTHON) { $env:LCORGNAV_PYTHON } else { 'python' }
$codeRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\code')).Path
$env:PYTHONPATH = $codeRoot
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
function L($m){ "$m $(Get-Date -f s)" | Add-Content p14.log }
function Fail($m){ L "FAIL $m"; exit 1 }
$man = Get-Content P14_FREEZE_MANIFEST.json -Raw | ConvertFrom-Json
foreach ($p in $man.files.PSObject.Properties) {
  $f = $p.Name
  if (-not [IO.Path]::IsPathRooted($f)) { $f = Join-Path $PSScriptRoot $f }
  if ((Get-FileHash $f -Algorithm SHA256).Hash.ToLower() -ne $p.Value) { Fail "hash mismatch $f" }
}
L 'START (hashes verified)'
foreach ($v in 'a','b','c') {
  foreach ($t in 'hard_confusable','scale300') {
    L "START train $v $t"
    & $py p14_train.py $v $t 1>> "train_${v}_${t}.log" 2>> "train_${v}_${t}.err"
    if ($LASTEXITCODE) { Fail "train $v $t $LASTEXITCODE" }
    L "EXIT train $v $t 0"
  }
}
$env:P14_FINAL = '1'
L 'START predict'
& $py p14_predict.py 1>> predict.log 2>> predict.err
if ($LASTEXITCODE) { Fail "predict $LASTEXITCODE" }
L 'EXIT predict 0 (hashed)'
"done $(Get-Date -f s)" | Set-Content P14_DONE.txt
L 'DONE'