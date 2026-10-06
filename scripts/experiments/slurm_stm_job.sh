#!/bin/bash
# ==============================================================================
# SLURM Worker Script for STM Experiments
#
# R is not installed on the cluster, so train_stm.R runs inside the lightweight
# STM image (Dockerfile.stm) and run_stm.py evaluates its outputs on the host.
#
# Arguments:
#   $1: INPUT_PREFIX     (e.g., fed, fed_stemmed, yelp_s10000)
#   $2: EXP_TARGET       (e.g., fed/fed_standard_stm)
#   $3: STM_IMAGE        (e.g., cast:stm-lite-v0.2.0)
#   $4: MODEL_IDX        (optional: integer run index for split mode)
#
# The image is loaded once per node from $HOME/docker_images/<image with ':'
# replaced by '_'>.tar.gz, e.g. cast_stm-lite-v0.2.0.tar.gz.
#
# Note: Any command-line resource flags passed to sbatch (e.g. --mem, --cpus-per-task,
# --time, --job-name) automatically override the fallback #SBATCH defaults below.
# ==============================================================================

#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

PROJECT_NAME="${PROJECT_NAME:-ca_bertopic}"

INPUT_PREFIX="${1:-$INPUT_PREFIX}"
EXP_TARGET="${2:-$EXP_TARGET}"
STM_IMAGE="${3:-$STM_IMAGE}"
MODEL_IDX="${4:-$MODEL_IDX}"

if [ -z "$INPUT_PREFIX" ] || [ -z "$EXP_TARGET" ] || [ -z "$STM_IMAGE" ]; then
    echo "ERROR: Missing required arguments. Usage: $0 <INPUT_PREFIX> <EXP_TARGET> <STM_IMAGE> [MODEL_IDX]" >&2
    exit 1
fi

echo "Job started at $(date) on $(hostname)"

# 1. Setup Job-Isolated SCRATCH Workspace & Cleanup Trap
JOB_SCRATCH_ROOT="$SCRATCH/job_${SLURM_JOB_ID}"
JOB_SCRATCH="${JOB_SCRATCH_ROOT}/${PROJECT_NAME}"

cleanup() {
    trap - EXIT INT TERM
    echo "Cleaning up temporary scratch directory: ${JOB_SCRATCH_ROOT}"
    cd "$HOME" || cd /tmp
    if [ -n "${JOB_SCRATCH_ROOT}" ] && [ -d "${JOB_SCRATCH_ROOT}" ]; then
        rm -rf "${JOB_SCRATCH_ROOT}"
        if [ ! -d "${JOB_SCRATCH_ROOT}" ]; then
            echo "Scratch directory successfully removed."
        else
            echo "Warning: Failed to completely remove ${JOB_SCRATCH_ROOT}."
        fi
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "${JOB_SCRATCH}"/{data/processed,results,models,logs,output,tables}

# 2. Sync Code base
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \
    --exclude='output/' --exclude='tables/' --exclude='.venv/' --exclude='.git/' \
    "$HOME/${PROJECT_NAME}/" "${JOB_SCRATCH}/"

cd "${JOB_SCRATCH}"

# 3. Sync the STM inputs for this job
rsync -a "$HOME/${PROJECT_NAME}/data/processed/${INPUT_PREFIX}_stm_data.rds" \
    "$HOME/${PROJECT_NAME}/data/processed/${INPUT_PREFIX}_bow.parquet" \
    data/processed/ || { echo "ERROR: STM inputs for ${INPUT_PREFIX} not found" >&2; exit 1; }

# 4. Load the STM image once per node (PCAD stores images per user and node)
if command -v podman >/dev/null 2>&1; then
    CONTAINER_CMD="podman"
else
    CONTAINER_CMD="docker"
fi
IMAGE_TAR="$HOME/docker_images/${STM_IMAGE//:/_}.tar.gz"

(
    flock -w 1800 9 || { echo "ERROR: Timed out waiting for the image load lock" >&2; exit 1; }
    if ! "$CONTAINER_CMD" image inspect "$STM_IMAGE" >/dev/null 2>&1; then
        echo "Loading ${STM_IMAGE} from ${IMAGE_TAR}..."
        "$CONTAINER_CMD" load -i "$IMAGE_TAR" || exit 1
    fi
) 9>"$SCRATCH/.stm_image_load.lock"
if [ $? -ne 0 ]; then
    echo "ERROR: Could not load ${STM_IMAGE} on $(hostname)" >&2
    exit 1
fi

# 5. Export UV path and environment configuration
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE="copy"
export GIT_COMMIT_REV="$(git -C "$HOME/${PROJECT_NAME}" rev-parse HEAD 2>/dev/null || echo "unknown")"
export GIT_DIRTY="$(git -C "$HOME/${PROJECT_NAME}" status --porcelain 2>/dev/null | grep -q . && echo "true" || echo "false")"

if [ -d "$HOME/${PROJECT_NAME}/.venv" ]; then
    ln -sfn "$HOME/${PROJECT_NAME}/.venv" "${JOB_SCRATCH}/.venv"
    UV_SYNC_OPT="--no-sync"
else
    UV_SYNC_OPT=""
fi

# 6. Train in the container, evaluate on the host
MODEL_ARGS=()
if [ -n "$MODEL_IDX" ] && [ "$MODEL_IDX" != "all" ] && [ "$MODEL_IDX" != "-" ]; then
    MODEL_ARGS=(--model "${MODEL_IDX}")
fi
uv run ${UV_SYNC_OPT} python scripts/experiments/run_stm.py --exp "${EXP_TARGET}" \
    "${MODEL_ARGS[@]}" --r-runner docker --docker-cmd "${CONTAINER_CMD}" --image "${STM_IMAGE}"

RUN_EXIT=$?
if [ $RUN_EXIT -ne 0 ]; then
    echo "ERROR: Experiment execution failed with exit code $RUN_EXIT" >&2
fi

# 7. Sync results back to HOME/slurm
mkdir -p "$HOME/slurm"/{results,logs,output,tables,models}
rsync -a "${JOB_SCRATCH}/results/" "$HOME/slurm/results/"
rsync -a "${JOB_SCRATCH}/logs/" "$HOME/slurm/logs/"
rsync -a "${JOB_SCRATCH}/output/" "$HOME/slurm/output/"
rsync -a "${JOB_SCRATCH}/tables/" "$HOME/slurm/tables/"
rsync -a "${JOB_SCRATCH}/models/" "$HOME/slurm/models/"

echo "Job finished at $(date)"
exit "$RUN_EXIT"
