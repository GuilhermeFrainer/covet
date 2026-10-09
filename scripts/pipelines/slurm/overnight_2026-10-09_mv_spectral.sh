#!/usr/bin/env bash
# Overnight batch of 2026-10-09: MV Spectral (COVET_CT) on the rebuilt Fed
# and Yelp, the last T1 gap. One run per SLURM job (30 jobs).
#
# Runs are ordered seed by seed, alternating Fed and Yelp, so if the night is
# too short the finished runs still cover every topic count (10-50) for one
# or two seeds on both datasets. Run indices: k is the outer loop and seeds
# the inner one, so seed 1 = runs 1,4,7,10,13; seed 2 = 2,5,...; seed 3 = 3,6,...
#
# Copy to ~/slurm/scripts/ and run from ~/slurm inside tmux; each batch keeps
# submitting as QOS slots free up, so the script must stay alive:
#   tmux new -s mvspec
#   cd ~/slurm
#   bash scripts/overnight_2026-10-09_mv_spectral.sh 2>&1 | tee -a logs/overnight_mv_spectral.log
# Detach with Ctrl-b d. Preview with --dry-run. FROM=N skips batches before N.

set -uo pipefail

QUEUE="$HOME/${PROJECT_NAME:-ca_bertopic}/scripts/pipelines/slurm/queue_exp.sh"
RESERVATION="${RESERVATION:-res-gdsfrainer-cidia}"
EXTRA=(--reservation "$RESERVATION" "$@")
FROM="${FROM:-1}"
N=0
mkdir -p logs

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

for runs in 1,4,7,10,13 2,5,8,11,14 3,6,9,12,15; do
    for dataset in fed yelp; do
        batch -d "$dataset" -e mv_spectral --split --runs "$runs" --time 12:00:00
    done
done

echo "=== $(date '+%F %T') all batches submitted"
