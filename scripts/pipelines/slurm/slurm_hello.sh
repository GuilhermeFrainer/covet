#!/bin/bash
#SBATCH --job-name=stm_image_check
#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

# ==============================================================================
# Checks that a compute node can load and run the STM image.
# Usage: sbatch scripts/pipelines/slurm/slurm_hello.sh [IMAGE]
# ==============================================================================
STM_IMAGE="${1:-cast:stm-lite-v0.2.0}"
IMAGE_TAR="$HOME/docker_images/${STM_IMAGE//:/_}.tar.gz"

echo "Job started at $(date) on $(hostname)"

if command -v podman >/dev/null 2>&1; then
    CONTAINER_CMD="podman"
else
    CONTAINER_CMD="docker"
fi
echo "Container CLI: $(command -v "$CONTAINER_CMD")"

if ! "$CONTAINER_CMD" image inspect "$STM_IMAGE" >/dev/null 2>&1; then
    echo "Loading ${STM_IMAGE} from ${IMAGE_TAR}..."
    "$CONTAINER_CMD" load -i "$IMAGE_TAR" || exit 1
fi

# The default command loads stm, data.table and optparse and prints sessionInfo().
"$CONTAINER_CMD" run --rm "$STM_IMAGE"

# Files written in a mounted directory must belong to the job's user.
mkdir -p "$SCRATCH/stm_image_check"
"$CONTAINER_CMD" run --rm -v "$SCRATCH/stm_image_check:/work" "$STM_IMAGE" \
    Rscript -e 'writeLines("ok", "/work/owner_check.txt")'
ls -ln "$SCRATCH/stm_image_check/owner_check.txt"
echo "Job user: $(id -u):$(id -g)"
rm -rf "$SCRATCH/stm_image_check"

echo "Job finished at $(date)"
