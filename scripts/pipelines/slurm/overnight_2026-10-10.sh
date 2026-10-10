#!/usr/bin/env bash
# Overnight batch of 2026-10-10: the runs still missing after the 2026-10-09
# batches, in two phases.
#
# Phase 1 (normal jobs, up to 5 at a time):
#   - Fed UMAP + Spectral runs 1-5, one run per job. On 2026-10-09 the pack
#     holding them hung on run 1 (k = 10) until its 24 h limit; a 1 h limit
#     per run now frees the slot if it hangs again.
#   - Weighted-Append sweep on Yelp and Trump-25k (w = 0, 0.05, 0.2, 0.3, 0.5).
#
# Phase 2 (starts only once you have no jobs left in SLURM):
#   - STM on Fed, Yelp, Trump-25k and Gadarian, all K, in ONE packed job.
#     On 2026-10-09 every STM run killed with exit 137 died about 11 s after
#     another of your jobs on the node ended, for both STM jobs at once. The
#     STM container runs under rootless Podman, outside the job's cgroup, so
#     the node's end-of-job cleanup most likely kills it as a stray process.
#     Running STM alone, in a single job, avoids any job ending beside it.
#     Do not submit anything else until this job has finished.
#     (STM has no --split here: counting STM runs imports polars, which
#     crashes on the login node.)
#
# Copy to ~/slurm/scripts/ and run from ~/slurm inside tmux:
#   tmux new -s night
#   cd ~/slurm
#   bash scripts/overnight_2026-10-10.sh 2>&1 | tee -a logs/overnight_2026-10-10.log
# Detach with Ctrl-b d. Preview with --dry-run. FROM=N skips batches before N.

set -uo pipefail

QUEUE="$HOME/${PROJECT_NAME:-ca_bertopic}/scripts/pipelines/slurm/queue_exp.sh"
RESERVATION="${RESERVATION:-res-gdsfrainer-cidia}"
EXTRA=(--reservation "$RESERVATION" "$@")
FROM="${FROM:-1}"
N=0
mkdir -p logs

DRY_RUN=false
for arg in "$@"; do
    [ "$arg" = "--dry-run" ] && DRY_RUN=true
done

# Submits one batch; a failed batch is reported and the next one still runs.
batch() {
    N=$((N + 1))
    if [ "$N" -lt "$FROM" ]; then
        echo "--- skipping batch $N (FROM=$FROM): queue_exp.sh $*"
        return
    fi
    echo "=== $(date '+%F %T') batch $N: queue_exp.sh $* ${EXTRA[*]}"
    bash "$QUEUE" "$@" "${EXTRA[@]}" -y || echo "!!! $(date '+%F %T') batch failed: $*"
}

# Waits until none of your jobs is queued or running.
wait_for_empty_queue() {
    if $DRY_RUN; then
        echo "--- (dry run) would wait here until your SLURM queue is empty"
        return
    fi
    while [ -n "$(squeue -u "$(whoami)" -h)" ]; do
        echo "    $(date '+%F %T') waiting: $(squeue -u "$(whoami)" -h | wc -l) job(s) still in SLURM"
        sleep 300
    done
    echo "=== $(date '+%F %T') queue empty"
}

# --- Phase 1 ---
# 1. Fed UMAP + Spectral, runs 1-5 (k = 10 x 3 seeds, k = 20 x 2 seeds).
batch -d fed -e umap_spectral --split --runs 1..5 --time 01:00:00

# 2. Weighted-Append sweep on Yelp and Trump-25k (w = 0.1 is already done).
batch -d yelp,trump_s25000 \
    -e append_umap_w000,append_umap_w005,append_umap_w020,append_umap_w030,append_umap_w050 \
    --pack 2 --time 08:00:00

# --- Phase 2 ---
echo "=== $(date '+%F %T') phase 1 submitted; STM waits for an empty queue"
wait_for_empty_queue

# 3. STM, all K, every dataset still missing runs, in one job.
batch -d fed,yelp,trump_s25000,gadarian -m stm --pack 4 --time 1-12:00:00

echo "=== $(date '+%F %T') all batches submitted. Submit nothing else until STM finishes."
