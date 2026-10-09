#!/usr/bin/env bash
# Overnight batch of 2026-10-08, after rebuilding Fed and Yelp with the
# chunking fixes. Re-runs the paper's models on both, fills the missing
# cells (Fed weighted-Append sweep, Trump-25k MV-HDBSCAN, Feature-Stacking
# and weighted Append) and redoes STM on the new chunks. See
# docs/paper_submission_checklist.md.
#
# Left out on purpose: the original TriTopic (FastTriTopic reproduces it
# about 15x faster) and MV Spectral on Fed and Yelp (days per model).
#
# On the cluster, copy it to ~/slurm/scripts/overnight.sh and run it from
# ~/slurm, so logs/ and the SLURM logs land there; the queue comes from the
# repository in ~/ca_bertopic. Run it inside tmux, because each batch keeps
# submitting until all its jobs are in SLURM:
#   tmux new -s overnight
#   cd ~/slurm
#   bash scripts/overnight.sh 2>&1 | tee -a logs/overnight.log
# Detach with Ctrl-b d; reattach with: tmux attach -t overnight
#
# Extra arguments go to every batch; preview everything first with:
#   bash scripts/overnight.sh --dry-run
# Jobs run in the reservation res-gdsfrainer-cidia; RESERVATION= overrides it.
# To resume after stopping it, FROM skips the batches before that number
# (already submitted batches would otherwise be submitted twice):
#   FROM=3 bash scripts/overnight.sh

set -uo pipefail

# The queue wrapper in the repository (~/ca_bertopic on the cluster); off the
# cluster, the copy next to this script.
QUEUE="$HOME/${PROJECT_NAME:-ca_bertopic}/scripts/pipelines/slurm/queue_exp.sh"
if [ ! -f "$QUEUE" ]; then
    QUEUE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/queue_exp.sh"
fi
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

# 1. Cheap HDBSCAN models on the rebuilt Fed and Yelp.
batch -d fed,yelp \
    -e baseline,append_umap,append_umap_w010,mv_hdbscan,feature_stacking_hdbscan \
    --pack 5 --time 12:00:00

# 2. Missing cells: the rest of the Fed weighted-Append sweep, and Trump-25k.
batch -d fed \
    -e append_umap_w000,append_umap_w005,append_umap_w020,append_umap_w030,append_umap_w050 \
    --pack 5 --time 12:00:00
batch -d trump_s25000 \
    -e append_umap_w010,mv_hdbscan,feature_stacking_hdbscan \
    --pack 3 --time 12:00:00

# 3. Slower models, one job per run so they spread over the running slots.
batch -d fed,yelp \
    -e aligned_umap,umap_spectral,mv_co_reg_spectral,fast_tritopic \
    --split --pack 5 --time 1-00:00:00

# 4. STM on the new chunks.
batch -d fed,yelp -m stm --split

echo "=== $(date '+%F %T') all batches submitted"
