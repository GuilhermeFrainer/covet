#!/bin/bash
#SBATCH --job-name=stm_docker_test
#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

# ==============================================================================
# IMAGE CONFIG
# ==============================================================================
IMAGE_NAME="cast"
VERSION="stm-lite-v0.1.0"

echo "Job started at $(date)"

# 1. Setup SCRATCH directory
# This avoids heavy I/O on the network-mounted $HOME directory
mkdir -p $SCRATCH/ca_bertopic
mkdir -p $SCRATCH/ca_bertopic/{data/processed,results,models,logs}

# 2. Sync CODE to SCRATCH (Excluding heavy data/results to be fast)
echo "Syncing code to SCRATCH..."
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \
    $HOME/ca_bertopic/ $SCRATCH/ca_bertopic/

# 3. Sync SPECIFIC DATA needed for the test
echo "Syncing Trump dataset..."
rsync -av $HOME/ca_bertopic/data/processed/trump_stm_data.rds \
    $SCRATCH/ca_bertopic/data/processed/

# 4. Ensure Docker image is loaded
if ! docker image inspect ${IMAGE_NAME}:${VERSION} >/dev/null 2>&1; then
    echo "Loading Docker image from $HOME/docker_images/..."
    docker load < $HOME/docker_images/${IMAGE_NAME}_${VERSION}.tar
fi

# 5. Run via DOCKER with VOLUME MOUNTS
# We mount the project into a SUBDIRECTORY to avoid shadowing /app/renv/library
# We set RENV_PATHS_LIBRARY to point to the libraries we built into the image
echo "Running Docker container..."
docker run --rm \
    -v $SCRATCH/ca_bertopic:/app/ca_bertopic \
    -w /app/ca_bertopic \
    -e RENV_PATHS_LIBRARY=/app/renv/library \
    ${IMAGE_NAME}:${VERSION} \
    Rscript scripts/r_scripts/stm_smoke_test.R

# 6. Sync RESULTS back to $HOME
echo "Syncing results back to $HOME/slurm..."
mkdir -p $HOME/slurm/{results,models,logs,output,tables}
rsync -av $SCRATCH/ca_bertopic/results/ $HOME/slurm/results/
rsync -av $SCRATCH/ca_bertopic/models/  $HOME/slurm/models/
rsync -av $SCRATCH/ca_bertopic/logs/    $HOME/slurm/logs/
rsync -av $SCRATCH/ca_bertopic/output/  $HOME/slurm/output/
rsync -av $SCRATCH/ca_bertopic/tables/  $HOME/slurm/tables/

echo "Job finished at $(date)"
