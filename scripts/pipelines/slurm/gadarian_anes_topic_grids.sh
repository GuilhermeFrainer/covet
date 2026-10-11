#!/usr/bin/env bash
# Gadarian and ANES on their own topic-count grids (Gadarian k = 4, 6, 8, 10,
# 12; ANES k = 8, 11, 14, 17, 20; see src/topic_grids.py): every primary model
# and every K-Means variant, 3 seeds each. About 22-26 h of compute, so 5-6 h
# with 5 jobs running at a time.
#
# Phase 1 (normal jobs, up to 5 at a time):
#   1. The 25 models other than MV Spectral and STM, 4 per packed job. The
#      longest pack (ANES TriTopic, UMAP + Spectral, Aligned UMAP MV K-Means)
#      takes about 4 h.
#   2. MV Spectral and MV Spectral (info_view=0), one run per job: they are
#      ~75% of the compute, up to ~22 min per ANES run.
#
# Phase 2 (starts only once you have no jobs left in SLURM):
#   3. STM on both datasets, all K, in ONE job. As on 2026-10-10, STM runs
#      alone: its rootless Podman container runs outside the job's cgroup and
#      is killed (exit 137) when another of your jobs on the node ends.
#      Do not submit anything else until it has finished.
#
# Needs a checkout with commit bae08af (the new topic counts) in
# ~/ca_bertopic. Do not start while the STM job of overnight_2026-10-10.sh is
# still running.
#
# Copy to ~/slurm/scripts/ and run from ~/slurm inside tmux:
#   tmux new -s grids
#   cd ~/slurm
#   bash scripts/gadarian_anes_topic_grids.sh 2>&1 | tee -a logs/gadarian_anes_topic_grids.log
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

MODELS=(
    # HDBSCAN family
    baseline aligned_umap append_umap
    append_umap_w000 append_umap_w005 append_umap_w010
    append_umap_w020 append_umap_w030 append_umap_w050
    feature_stacking_hdbscan mv_hdbscan
    # Spectral family (MV Spectral runs in batch 2)
    umap_spectral mv_co_reg_spectral
    # K-Means family
    k_means mv_k_means mv_spherical_k_means
    pca_k_means pca_mv_k_means pca_mv_spherical_k_means
    aligned_umap_mv_k_means aligned_umap_mv_spherical_k_means
    append_umap_mv_k_means append_umap_mv_spherical_k_means
    # External baselines (STM runs in batch 3)
    tritopic fast_tritopic
)
MODEL_LIST=$(IFS=,; echo "${MODELS[*]}")

# --- Phase 1 ---
# 1. Everything but MV Spectral and STM, 4 models per job.
batch -d gadarian,anes -e "$MODEL_LIST" --pack 4 --time 24:00:00

# 2. MV Spectral, one run (k x seed) per job.
batch -d gadarian,anes -e mv_spectral,mv_spectral_info0 --split --time 06:00:00

# --- Phase 2 ---
echo "=== $(date '+%F %T') phase 1 submitted; STM waits for an empty queue"
wait_for_empty_queue

# 3. STM, all K on both datasets, in one job.
batch -d gadarian,anes -e stm --pack 2 --time 06:00:00

echo "=== $(date '+%F %T') all batches submitted. Submit nothing else until STM finishes."
