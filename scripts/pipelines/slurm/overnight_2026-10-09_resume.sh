#!/usr/bin/env bash
# Resumes the overnight batch of 2026-10-08 (overnight_2026-10-08.sh) with
# only what it failed to produce:
#
# - MV-HDBSCAN and Feature-Stacking on Fed, Yelp and Trump-25k. They crashed
#   with "No module named 'mv_hdbscan'": the cluster's .venv lacked the local
#   MV-HDBSCAN package. This script installs it first if it is missing.
# - STM on Fed and Yelp. Batch 5 never submitted: `--split` imports polars,
#   which dies with an illegal instruction on the login node (no AVX2). One
#   job per dataset runs all five K values in sequence instead.
#
# Everything else either finished or is still running; do not resubmit it.
#
# Copy to ~/slurm/scripts/ and run from ~/slurm inside tmux:
#   tmux new -s resume
#   cd ~/slurm
#   bash scripts/overnight_2026-10-09_resume.sh 2>&1 | tee -a logs/overnight_resume.log
# Preview with --dry-run. FROM=2 skips batch 1 when resuming.

set -uo pipefail

REPO="$HOME/${PROJECT_NAME:-ca_bertopic}"
QUEUE="$REPO/scripts/pipelines/slurm/queue_exp.sh"
RESERVATION="${RESERVATION:-res-gdsfrainer-cidia}"
EXTRA=(--reservation "$RESERVATION" "$@")
FROM="${FROM:-1}"
N=0
mkdir -p logs

DRY_RUN=false
for arg in "$@"; do
    [ "$arg" = "--dry-run" ] && DRY_RUN=true
done

# MV-HDBSCAN must import from the shared .venv the jobs link to.
if ! "$REPO/.venv/bin/python" -c "import mv_hdbscan" 2>/dev/null; then
    if $DRY_RUN; then
        echo "!!! mv_hdbscan is missing from $REPO/.venv (a real run installs it)"
    else
        echo "=== Installing MV-HDBSCAN into $REPO/.venv (--no-deps leaves the rest alone)"
        (cd "$REPO" && "$HOME/.local/bin/uv" pip install --no-deps "$HOME/MV-HDBSCAN")
        if ! "$REPO/.venv/bin/python" -c "import mv_hdbscan"; then
            echo "!!! mv_hdbscan still does not import; not submitting anything."
            exit 1
        fi
    fi
fi
echo "=== mv_hdbscan check done"

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

# 1. MV-HDBSCAN and Feature-Stacking (one job per dataset, both models).
batch -d fed,yelp,trump_s25000 \
    -e mv_hdbscan,feature_stacking_hdbscan \
    --pack 2 --time 12:00:00

# 2. STM on the new chunks; no --split (see above).
batch -d fed,yelp -m stm

echo "=== $(date '+%F %T') all batches submitted"
