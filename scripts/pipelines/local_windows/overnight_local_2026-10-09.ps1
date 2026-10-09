# Overnight local batch of 2026-10-09: the quick runs that the cluster's
# overnight batch (scripts/pipelines/slurm/overnight_2026-10-08.sh) does not
# submit. See missing_runs_2026-10-09.xlsx for the full checklist.
#
# Each run is started on its own (`--model N`), so stopping the script loses
# at most the run in progress. Every run prints a banner with its entry and
# run number; to resume, comment out the finished entries in $Jobs below and
# shorten the current entry's Runs (e.g. 1..15 -> 7..15).
#
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/pipelines/local_windows/overnight_local_2026-10-09.ps1
# Finished and failed runs are also listed in logs/overnight_local_<timestamp>.log.

$ErrorActionPreference = "Continue"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $RepoRoot

# Ordered shortest first, so an early stop still clears whole cells.
# Runner "opt" = run_optimizer.py, "stm" = run_stm.py (local R).
# Neural configs: runs 1..15 = k 10, 20, 30, 40, 50 x 3 seeds (seeds vary fastest).
# STM configs: runs 1..5 = k 10, 20, 30, 40, 50.
$Jobs = @(
    # --- 1. Gadarian STM, the missing k = 30 (~1 min) ---
    @{ Runner = "stm"; Exp = "gadarian/gadarian_standard_stm"; Runs = 3..3 }

    # --- 2. PCA K-Means variants, catalog only, not in the paper (~15 min) ---
    @{ Runner = "opt"; Exp = "fed/fed_standard_pca_k_means"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "fed/fed_standard_pca_mv_k_means"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "fed/fed_standard_pca_mv_spherical_k_means"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "yelp/yelp_standard_pca_k_means"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "yelp/yelp_standard_pca_mv_k_means"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "yelp/yelp_standard_pca_mv_spherical_k_means"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "trump_s25000/trump_s25000_standard_pca_mv_k_means"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "trump_s25000/trump_s25000_standard_pca_mv_spherical_k_means"; Runs = 1..15 }

    # --- 3. Trump-25k STM (T2; ~1-2 h) ---
    @{ Runner = "stm"; Exp = "trump_s25000/trump_s25000_standard_stm"; Runs = 1..5 }

    # --- 4. Yelp weighted-Append sweep (F1; ~2 min/run, ~2.5 h) ---
    @{ Runner = "opt"; Exp = "yelp/yelp_standard_append_umap_w000"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "yelp/yelp_standard_append_umap_w005"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "yelp/yelp_standard_append_umap_w020"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "yelp/yelp_standard_append_umap_w030"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "yelp/yelp_standard_append_umap_w050"; Runs = 1..15 }

    # --- 5. Trump-25k weighted-Append sweep (F1; ~4 min/run, ~5 h) ---
    @{ Runner = "opt"; Exp = "trump_s25000/trump_s25000_standard_append_umap_w000"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "trump_s25000/trump_s25000_standard_append_umap_w005"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "trump_s25000/trump_s25000_standard_append_umap_w020"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "trump_s25000/trump_s25000_standard_append_umap_w030"; Runs = 1..15 }
    @{ Runner = "opt"; Exp = "trump_s25000/trump_s25000_standard_append_umap_w050"; Runs = 1..15 }
)

$Scripts = @{
    opt = "scripts/experiments/run_optimizer.py"
    stm = "scripts/experiments/run_stm.py"
}

New-Item -ItemType Directory -Force "logs" | Out-Null
$Log = "logs/overnight_local_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$Total = ($Jobs | ForEach-Object { @($_.Runs).Count } | Measure-Object -Sum).Sum
$Done = 0
$Failed = @()
$Line = "=" * 78

function Write-Log([string]$Message) {
    $stamped = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    Add-Content -Path $Log -Value $stamped
    return $stamped
}

Write-Host $Line -ForegroundColor Cyan
Write-Host "Local overnight batch: $($Jobs.Count) entries, $Total runs. Log: $Log" -ForegroundColor Cyan
Write-Host $Line -ForegroundColor Cyan
Write-Log "START: $($Jobs.Count) entries, $Total runs" | Out-Null

for ($j = 0; $j -lt $Jobs.Count; $j++) {
    $job = $Jobs[$j]
    $runs = @($job.Runs)
    for ($r = 0; $r -lt $runs.Count; $r++) {
        $n = $runs[$r]
        $Done++
        $cmd = "uv run python $($Scripts[$job.Runner]) --exp $($job.Exp) --model $n"
        $remaining = if ($r + 1 -lt $runs.Count) { "$($runs[$r + 1])..$($runs[-1])" } else { "none (comment the entry out)" }

        Write-Host ""
        Write-Host $Line -ForegroundColor Yellow
        Write-Host ("[{0}/{1}]  {2}" -f $Done, $Total, (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor Yellow
        Write-Host "  ENTRY : $($j + 1) of $($Jobs.Count)   $($job.Exp)" -ForegroundColor Yellow
        Write-Host "  RUN   : $n   (this entry: $($r + 1) of $($runs.Count))" -ForegroundColor Yellow
        Write-Host "  CMD   : $cmd" -ForegroundColor Yellow
        Write-Host "  IF YOU STOP NOW: comment out entries before $($j + 1); set this entry's Runs = $n..$($runs[-1])" -ForegroundColor Yellow
        Write-Host $Line -ForegroundColor Yellow

        $start = Get-Date
        & uv run python $Scripts[$job.Runner] --exp $job.Exp --model $n
        $code = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)

        if ($code -eq 0) {
            $msg = Write-Log "OK      $($job.Exp) run $n ($mins min). Next for this entry: $remaining"
            Write-Host $msg -ForegroundColor Green
        } else {
            $Failed += "$($job.Exp) run $n"
            $msg = Write-Log "FAILED  $($job.Exp) run $n (exit $code, $mins min)"
            Write-Host $msg -ForegroundColor Red
        }
    }
    $msg = Write-Log "ENTRY DONE  $($j + 1): $($job.Exp)  -> safe to comment out"
    Write-Host $msg -ForegroundColor Green
}

Write-Host ""
Write-Host $Line -ForegroundColor Cyan
Write-Host (Write-Log "FINISHED: $Done runs, $($Failed.Count) failed") -ForegroundColor Cyan
foreach ($f in $Failed) { Write-Host "  FAILED: $f" -ForegroundColor Red }
Write-Host "Next: copy the cluster's raw CSVs into results/, then run scripts/analysis/merge_results.py" -ForegroundColor Cyan
Write-Host $Line -ForegroundColor Cyan
