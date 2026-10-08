#!/bin/bash
# ==============================================================================
# SLURM Worker Script for Packed Experiments
#
# Runs several worker invocations one after another inside a single SLURM job,
# so a batch fits under the per-user submit limit. Each invocation is a worker
# script (slurm_job.sh or slurm_stm_job.sh) followed by its arguments, and
# invocations are separated by '::':
#
#   slurm_pack_job.sh slurm_job.sh fed fed/fed_standard_baseline --remove-rep-stopwords \
#       :: slurm_stm_job.sh fed fed/fed_standard_stm cast:stm-lite-v0.2.0
#
# Each worker sets up and removes its own scratch workspace and copies its
# results back before the next one starts, so finished runs survive a later
# failure or the job reaching its time limit. A failed run does not stop the
# rest; the job exits non-zero if any run failed.
#
# Note: Any command-line resource flags passed to sbatch (e.g. --mem, --cpus-per-task,
# --time, --job-name) automatically override the fallback #SBATCH defaults below.
# ==============================================================================

#SBATCH --partition=cidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.err

SEPARATOR="::"

if [ $# -eq 0 ]; then
    echo "ERROR: No runs given. Usage: $0 <WORKER> [ARGS...] [:: <WORKER> [ARGS...]]..." >&2
    exit 1
fi

N_RUNS=1
for arg in "$@"; do
    [ "$arg" = "$SEPARATOR" ] && N_RUNS=$((N_RUNS + 1))
done

echo "Packed job started at $(date) on $(hostname) with ${N_RUNS} runs"

RUN=0
FAILED=()
CURRENT=()

run_current() {
    RUN=$((RUN + 1))
    if [ ${#CURRENT[@]} -eq 0 ]; then
        echo "ERROR: Run ${RUN}/${N_RUNS} is empty" >&2
        FAILED+=("${RUN}")
        return
    fi
    echo "================================================================="
    echo " Run ${RUN}/${N_RUNS} started at $(date): ${CURRENT[*]:1}"
    echo "================================================================="
    bash "${CURRENT[@]}"
    local exit_code=$?
    echo " Run ${RUN}/${N_RUNS} finished at $(date) with exit code ${exit_code}"
    if [ $exit_code -ne 0 ]; then
        FAILED+=("${RUN} (${CURRENT[*]:1})")
    fi
}

for arg in "$@"; do
    if [ "$arg" = "$SEPARATOR" ]; then
        run_current
        CURRENT=()
    else
        CURRENT+=("$arg")
    fi
done
run_current

echo "================================================================="
echo "Packed job finished at $(date): $((N_RUNS - ${#FAILED[@]}))/${N_RUNS} runs succeeded"
if [ ${#FAILED[@]} -gt 0 ]; then
    for failed in "${FAILED[@]}"; do
        echo "  Failed run ${failed}" >&2
    done
    exit 1
fi
exit 0
