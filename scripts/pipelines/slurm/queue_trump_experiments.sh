#!/bin/bash

# ==============================================================================
# Master Script to queue all standard Trump experiments on SLURM
# Generates one separate job for each model instance index (1 to 15).
# ==============================================================================

# Models to run
MODELS=("aligned_umap" "append_umap" "baseline" "fast_tritopic" "mv_co_reg_spectral" "mv_spectral" "mv_spectral_info0" "tritopic" "umap_spectral")

# Ensure slurm log directory exists
mkdir -p slurm_log

# 1. Queue CA-BERTopic experiments (Non-STM)
# Standard configurations (nr_topics = 10, 20, 30, 40, 50) and 3 seeds (15 total runs)
for model in "${MODELS[@]}"; do
    for model_idx in {1..15}; do
        job_name="ca_bertopic_trump_${model}_m${model_idx}"
        echo "Queuing job: $job_name"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

echo "Job started at \$(date) on \$(hostname)"

# 1. Setup Job-Isolated SCRATCH Workspace & Cleanup Trap
JOB_SCRATCH_ROOT="\$SCRATCH/ca_bertopic_\${SLURM_JOB_ID}"
JOB_SCRATCH="\${JOB_SCRATCH_ROOT}/ca_bertopic"

cleanup() {
    trap - EXIT INT TERM
    echo "Cleaning up temporary scratch directory: \${JOB_SCRATCH_ROOT}"
    cd "\$HOME" || cd /tmp
    if [ -n "\${JOB_SCRATCH_ROOT}" ] && [ -d "\${JOB_SCRATCH_ROOT}" ]; then
        rm -rf "\${JOB_SCRATCH_ROOT}"
        if [ ! -d "\${JOB_SCRATCH_ROOT}" ]; then
            echo "Scratch directory successfully removed."
        else
            echo "Warning: Failed to completely remove \${JOB_SCRATCH_ROOT}."
        fi
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "\${JOB_SCRATCH}"/{data/processed,results,models,logs,output,tables}

# 2. Sync Code base
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \
    --exclude='output/' --exclude='tables/' --exclude='.venv/' --exclude='.git/' \
    \$HOME/ca_bertopic/ "\${JOB_SCRATCH}/"

FAST_TRITOPIC_SRC=""
if [ -d "\$HOME/fast-tritopic" ]; then
    FAST_TRITOPIC_SRC="\$HOME/fast-tritopic"
elif [ -d "\$HOME/fast_tritopic" ]; then
    FAST_TRITOPIC_SRC="\$HOME/fast_tritopic"
fi

if [ -n "\$FAST_TRITOPIC_SRC" ]; then
    mkdir -p "\${JOB_SCRATCH_ROOT}/fast-tritopic"
    rsync -avL --exclude='.venv/' --exclude='.git/' \
        "\$FAST_TRITOPIC_SRC/" "\${JOB_SCRATCH_ROOT}/fast-tritopic/"
fi

cd "\${JOB_SCRATCH}"

# 3. Sync specific data file (Trump embeddings)
rsync -a \$HOME/ca_bertopic/data/processed/trump_embeddings.parquet "\${JOB_SCRATCH}/data/processed/"

# 4. Export UV path and environment configuration
export PATH="\$HOME/.local/bin:\$PATH"
export UV_LINK_MODE="copy"

# Link pre-built virtual environment from HOME if available to avoid
# 10GB package copying and concurrent rebuild race conditions on scratch
if [ -d "\$HOME/ca_bertopic/.venv" ]; then
    ln -sfn "\$HOME/ca_bertopic/.venv" "\${JOB_SCRATCH}/.venv"
    UV_SYNC_OPT="--no-sync"
else
    UV_SYNC_OPT=""
fi

# 5. Run specific model instance
uv run \${UV_SYNC_OPT} python scripts/experiments/run_optimizer.py --exp trump/trump_standard_${model} --model ${model_idx}
RUN_EXIT=\$?
if [ \$RUN_EXIT -ne 0 ]; then
    echo "ERROR: Experiment execution failed with exit code \$RUN_EXIT" >&2
    exit \$RUN_EXIT
fi

# 6. Sync results back to HOME/slurm
mkdir -p \$HOME/slurm/{results,logs,output,tables,models}
rsync -a "\${JOB_SCRATCH}/results/" \$HOME/slurm/results/
rsync -a "\${JOB_SCRATCH}/logs/" \$HOME/slurm/logs/
rsync -a "\${JOB_SCRATCH}/output/" \$HOME/slurm/output/
rsync -a "\${JOB_SCRATCH}/tables/" \$HOME/slurm/tables/
rsync -a "\${JOB_SCRATCH}/models/" \$HOME/slurm/models/

echo "Job finished at \$(date)"
EOF
    done
done

# 2. Queue STM experiments via Docker
# For Trump STM standard, K values are 10, 20, 30, 40, 50 and 3 seeds
K_VALUES=(10 20 30 40 50)
SEEDS=(36201624 62613654 57116123)
IMAGE_NAME="cast"
VERSION="stm-lite-v0.1.0"
FORMULA="~ log(favorites + 1) + log(retweets + 1) + date + as.factor(device) + as.factor(is_retweet) + as.factor(is_deleted) + as.factor(is_flagged)"

for k in "${K_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        model_id="stm_k${k}_seed${seed}"
        job_name="stm_trump_k${k}_s${seed}"
        echo "Queuing job: $job_name"

        output_dir="results/trump_${model_id}"
        model_path="models/trump_${model_id}.rds"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

echo "Job started at \$(date) on \$(hostname)"

# 1. Setup Job-Isolated SCRATCH Workspace & Cleanup Trap
JOB_SCRATCH="\$SCRATCH/ca_bertopic_\${SLURM_JOB_ID}"

cleanup() {
    trap - EXIT INT TERM
    echo "Cleaning up temporary scratch directory: \${JOB_SCRATCH}"
    cd "\$HOME" || cd /tmp
    if [ -n "\${JOB_SCRATCH}" ] && [ -d "\${JOB_SCRATCH}" ]; then
        rm -rf "\${JOB_SCRATCH}"
        if [ ! -d "\${JOB_SCRATCH}" ]; then
            echo "Scratch directory successfully removed."
        else
            echo "Warning: Failed to completely remove \${JOB_SCRATCH}."
        fi
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "\${JOB_SCRATCH}"/{data/processed,results,models,logs}

# 2. Sync Code base
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \
    --exclude='.venv/' --exclude='.git/' \
    \$HOME/ca_bertopic/ "\${JOB_SCRATCH}/"

# 3. Sync specific data file (Trump STM data)
rsync -a \$HOME/ca_bertopic/data/processed/trump_stm_data.rds "\${JOB_SCRATCH}/data/processed/"

# 4. Ensure Docker image is loaded
if ! docker image inspect ${IMAGE_NAME}:${VERSION} >/dev/null 2>&1; then
    docker load < \$HOME/docker_images/${IMAGE_NAME}_${VERSION}.tar
fi

# 5. Run training via Docker
mkdir -p "\${JOB_SCRATCH}/${output_dir}"
docker run --rm \
    -v "\${JOB_SCRATCH}:/app/ca_bertopic" \
    -w /app/ca_bertopic \
    -e RENV_PATHS_LIBRARY=/app/renv/library \
    ${IMAGE_NAME}:${VERSION} \
    Rscript scripts/r_scripts/train_stm.R \
    --rds_path "data/processed/trump_stm_data.rds" \
    --k "${k}" \
    --output_dir "${output_dir}" \
    --seed "${seed}" \
    --model_path "${model_path}" \
    --prevalence_formula "${FORMULA}"

# 6. Sync results back to HOME/slurm
mkdir -p \$HOME/slurm/{results,models,logs}
rsync -a "\${JOB_SCRATCH}/${output_dir}/" \$HOME/slurm/results/${model_id}/
rsync -a "\${JOB_SCRATCH}/${model_path}"   \$HOME/slurm/models/
rsync -a "\${JOB_SCRATCH}/logs/"           \$HOME/slurm/logs/

echo "Job finished at \$(date)"
EOF
    done
done

echo "------------------------------------------------"
echo "All Trump jobs have been dispatched to the scheduler."
