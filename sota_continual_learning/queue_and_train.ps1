<#
  queue_and_train.ps1
  Waits for the RTX 5090 to have enough free VRAM, then launches the 50k-step
  continuous training run. Decoupled from any interactive session so it fires
  whenever the currently-running jobs (4000-step run + monte-carlo) finish.

  Everything is logged: a queue status file while waiting, and a timestamped
  training log once the run starts.
#>
param(
  [int]$TotalSteps   = 50000,
  [int]$EvalInterval = 2500,
  [int]$BatchSize    = 2,
  [int]$ContextWindow = 168,
  [int]$NeedFreeMiB  = 29500,   # > run_production's 28672 MiB admission requirement, with margin
  [int]$MaxWaitHours = 12
)

$ErrorActionPreference = 'Continue'
$root    = if ($env:MBAL_PROJECT_ROOT) { $env:MBAL_PROJECT_ROOT } else { (Resolve-Path "$PSScriptRoot\..").Path }
$outDir  = Join-Path $root 'sota_continual_learning\output_production'
$stamp   = Get-Date -Format 'yyyyMMdd-HHmmss'
$trainLog = Join-Path $outDir "run_50k_$stamp.log"
$queueLog = Join-Path $outDir 'queue_status.txt'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Location $root

function Get-FreeMiB {
  $v = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
  if ($v) { return [int]($v.Trim() -split "`n")[0] }
  return 0
}
function Write-Queue([string]$msg) {
  $line = "{0} | {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
  Set-Content -LiteralPath $queueLog -Value $line -Encoding UTF8
}
function Get-GpuBlockers {
  # Other GPU training jobs whose VRAM oscillates (monte-carlo) or competing runs.
  # We wait for these PIDs to fully EXIT, not just for a transient VRAM dip.
  return @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessId -ne $PID -and (
      $_.CommandLine -match 'monte_carlo' -or $_.CommandLine -match 'run_production\.py') })
}

# ---- Phase 1: wait for the GPU. Launch only when (a) no competing GPU job is
# running AND (b) free VRAM is stably above the admission requirement. The
# blocker-exit check defeats the monte-carlo's per-trial VRAM oscillation. ----
$deadline = (Get-Date).AddHours($MaxWaitHours)
$stableHits = 0
while ($true) {
  $blockers = Get-GpuBlockers
  $free = Get-FreeMiB
  if ($blockers.Count -eq 0 -and $free -ge $NeedFreeMiB) {
    $stableHits++
    Write-Queue "READY: $free MiB free, no competing GPU jobs [stable $stableHits/3]"
    if ($stableHits -ge 3) { break }   # three consecutive clean reads -> launch
  } else {
    $stableHits = 0
    $ids = ($blockers | ForEach-Object { $_.ProcessId }) -join ','
    Write-Queue "WAITING: $free MiB free; blockers=[$ids] (need free>=$NeedFreeMiB and 0 blockers)"
  }
  if ((Get-Date) -gt $deadline) {
    Write-Queue "ABORTED: waited past $MaxWaitHours h without a clear GPU."
    exit 2
  }
  Start-Sleep -Seconds 30
}

# ---- Phase 2: launch the continuous run ----
Write-Queue "LAUNCHING 50k run -> $trainLog"
"=== queue_and_train launching $stamp | steps=$TotalSteps batch=$BatchSize ctx=$ContextWindow ===" |
  Out-File -LiteralPath $trainLog -Encoding UTF8

& python sota_continual_learning\run_production.py `
    --total-steps $TotalSteps `
    --eval-interval $EvalInterval `
    --batch-size $BatchSize `
    --context-window $ContextWindow `
    --output-dir $outDir *>> $trainLog

$code = $LASTEXITCODE
Write-Queue "TRAINING PROCESS EXITED code=$code (see $trainLog)"
exit $code
