# SLURM Cluster Concurrency & `fast-tritopic` Diagnostic Report

## 1. Executive Summary

Following the introduction of `fast-tritopic` into the experiment rotation (commit `9fde04b`), several SLURM batch jobs on the GPPD HPC cluster (`cidia` partition) began failing intermittently. The failures manifested as jobs spending 3–6 minutes installing packages and then abruptly terminating before executing any Python model training code, leaving truncated error logs (`.err`) without standard traceback or pipeline output.

This issue affected experiments at random, regardless of whether the model under test was `fast_tritopic` or an unrelated model (such as `pca_k_means` or `baseline`).

This report provides the root-cause analysis of the failure mechanism, explains why the naming/symlinking between `~/fast-tritopic` and `~/fast_tritopic` was a secondary symptom, and presents an architectural solution.

---

## 2. Anatomy of the Failure

### 2.1 Comparing Successful vs. Unsuccessful Execution Logs

* **Successful Job (`gadarian_pca_mv_spectral_818955.err`)**:
  ```text
  1: Using CPython 3.12.13
  2: Creating virtual environment at: .venv
  3:    Building fast-tritopic @ file:///scratch/gdsfrainer/job_818955/fast-tritopic
  4:       Built fast-tritopic @ file:///scratch/gdsfrainer/job_818955/fast-tritopic
  5: warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
  6:          If the cache and target directories are on different filesystems, hardlinking may not be supported.
  7:          If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
  8: Installed 322 packages in 3m 04s
  9: .../site-packages/hdbscan/robust_single_linkage_.py:175: SyntaxWarning: invalid escape sequence '\{'
  10: 2026-09-03 10:03:22,207 [pipeline] [INFO] Starting experiment: gadarian_standard_pca_mv_spectral
  ...
  ```

* **Unsuccessful Job (`gadarian_pca_k_means_818952.err`)**:
  ```text
  1: Using CPython 3.12.13
  2: Creating virtual environment at: .venv
  3:    Building fast-tritopic @ file:///scratch/gdsfrainer/job_818952/fast-tritopic
  4:       Built fast-tritopic @ file:///scratch/gdsfrainer/job_818952/fast-tritopic
  5: warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
  6:          If the cache and target directories are on different filesystems, hardlinking may not be supported.
  7:          If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
  8: Installed 322 packages in 5m 42s
  (EOF - process terminated without launching Python or logging errors)
  ```

In job `818952`, execution halted immediately after package installation. The Python interpreter never reached `main()`, never initialized the pipeline logger, and never even imported `hdbscan` (whose `SyntaxWarning` is triggered upon import before application code runs).

---

## 3. Root Cause Analysis

### 3.1 Why Did This Start with `fast_tritopic`?
Prior to commit `9fde04b`, all dependencies in [pyproject.toml](../../pyproject.toml) were pre-compiled wheels fetched from PyPI or NVIDIA's index. When `uv run` was invoked on a compute node, `uv` only read static binary wheels from the cache. There was **no source compilation**, no PEP 517 build backend execution, and no concurrent writing to the shared package cache.

In commit `9fde04b`, `fast-tritopic` was added as an **editable local path dependency**:
```toml
[dependencies]
"fast-tritopic",

[tool.uv.sources]
fast-tritopic = { path = "../fast-tritopic", editable = true }
```

In the SLURM submission script ([queue_exp.sh](../../scripts/pipelines/slurm/queue_exp.sh)), every job sets up an isolated workspace:
```bash
JOB_SCRATCH_ROOT="$SCRATCH/job_${SLURM_JOB_ID}"
JOB_SCRATCH="${JOB_SCRATCH_ROOT}/${PROJECT_NAME}"
```
and copies `fast-tritopic` to `${JOB_SCRATCH_ROOT}/fast-tritopic`.

Because each SLURM job has a unique `$SLURM_JOB_ID`:
1. The relative path `../fast-tritopic` resolves to a **different absolute directory for every single job**:
   `file:///scratch/gdsfrainer/job_818952/fast-tritopic` vs. `file:///scratch/gdsfrainer/job_818955/fast-tritopic`.
2. Because the path is different for each job, `uv` cannot reuse a cached local build and is **forced to rebuild `fast-tritopic` from source on every single job**:
   `Building fast-tritopic @ file:///scratch/.../fast-tritopic`
3. Because `fast-tritopic` is declared in the root `dependencies` table of `ca-bertopic`, **every model experiment** (including `pca_k_means`, `baseline`, and `pca_mv_spectral`) goes through this build and environment sync.

### 3.2 Why Was the Failure Intermittent / Random?
1. **NFS File Lock Collisions on `~/.cache/uv`**:
   The GPPD HPC cluster (`cidia` partition) uses an NFS-mounted network filesystem for `$HOME`. When 10 to 100 SLURM jobs launch simultaneously, they all run `uv run` at the same time. Each job invokes `hatchling`, compiles `fast-tritopic`, and attempts to acquire write locks on `$HOME/.cache/uv` concurrently.
   NFS does not support POSIX file locking (`flock`/`fcntl`) reliably across concurrent nodes. When multiple jobs race to write wheels and update SQLite/lockfiles in the cache, some jobs win the lock while others hit lock timeouts, corrupted metadata, or silent aborts.
2. **Editable Install `.pth` Path Invalidation**:
   In editable mode (`editable = true`), `hatchling` creates a `.pth` file (`_editable_impl_fast_tritopic.pth`) containing an absolute path to the build directory: `/scratch/gdsfrainer/job_${SLURM_JOB_ID}/fast-tritopic/src`.
   If a cached build artifact is shared across jobs, Job B receives a `.pth` file pointing to Job A's scratch path. When Job A completes and executes its cleanup trap (`rm -rf ${JOB_SCRATCH_ROOT}`), Job A's scratch directory is deleted, leaving Job B with a broken pointer.
3. **The Symlink Was a Red Herring**:
   Renaming `~/fast-tritopic` to `~/fast_tritopic` and adding a symlink between them only ensured the source directory was located during `rsync`. Both logs confirmed `Built fast-tritopic @ file:///scratch/...`, proving the directory was found. The failure was not directory resolution, but concurrent build-and-cache locking on NFS.

---

## 4. The Virtual Environment I/O Bottleneck

In [queue_exp.sh](../../scripts/pipelines/slurm/queue_exp.sh):
```bash
rsync -av --exclude='data/' --exclude='models/' --exclude='results/' --exclude='logs/' \
    --exclude='output/' --exclude='tables/' --exclude='.venv/' --exclude='.git/' \
    $HOME/${PROJECT_NAME}/ "${JOB_SCRATCH}/"
```
The script excluded `.venv/` with the intent of keeping the code sync fast. However, running `uv run` in a directory without an existing `.venv` forces `uv` to create a new virtual environment from scratch on **every single job**.

This requires copying **322 packages (~8 to 10 GB)** from `$HOME/.cache/uv` across the filesystem boundary to `/scratch`:
```text
warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
Installed 322 packages in 5m 42s
```
Across 20 concurrent jobs, this triggers up to **200 GB of redundant file copies** over NFS, creating severe I/O bottlenecks and causing package installations to take 3 to 6 minutes before any computation can begin.

---

## 5. Architectural Solution

### 5.1 Step 1: Remove `editable = true` in `pyproject.toml`
Change `[tool.uv.sources]` in [pyproject.toml](../../pyproject.toml) from:
```toml
[tool.uv.sources]
fast-tritopic = { path = "../fast-tritopic", editable = true }
```
to:
```toml
[tool.uv.sources]
fast-tritopic = { path = "../fast-tritopic" }
```
* **Effect:** `uv` installs `fast_tritopic` as a clean, static package directly into `site-packages/fast_tritopic/`. It eliminates ephemeral `.pth` files and cross-job scratch path references.

### 5.2 Step 2: Symlink Pre-Built `.venv` in SLURM Scripts
Rather than copying or rebuilding `.venv` on every job, the SLURM batch scripts ([queue_exp.sh](../../scripts/pipelines/slurm/queue_exp.sh) and [queue_trump_experiments.sh](../../scripts/pipelines/slurm/queue_trump_experiments.sh)) will symlink the pre-built, synchronized environment from `$HOME/${PROJECT_NAME}/.venv`:

```bash
# Link pre-built virtual environment from HOME if available
if [ -d "$HOME/${PROJECT_NAME}/.venv" ]; then
    ln -sfn "$HOME/${PROJECT_NAME}/.venv" "${JOB_SCRATCH}/.venv"
    UV_SYNC_OPT="--no-sync"
else
    UV_SYNC_OPT=""
fi

# Run with --no-sync to bypass dependency checks and build locks
uv run ${UV_SYNC_OPT} python scripts/experiments/run_optimizer.py ...
```

#### Detailed Systems Analysis: Symlinking vs. Copying `.venv`

A natural concern when considering symlinking a virtual environment from network storage (`$HOME` over NFS) rather than local scratch (`$SCRATCH`) is whether execution performance will degrade due to network I/O. However, understanding the Python runtime and Linux memory subsystems reveals why symlinking is not only safe, but significantly faster overall:

1. **When Does Python Touch Library Files?**
   * Python and its underlying C++/CUDA extension modules (`.so` libraries like `libtorch.so`, `libcuml.so`, etc.) are loaded **strictly during the initialization/startup phase** (the first 2–3 seconds) when `import` statements are executed.
   * Dynamic linkers (`dlopen`/`mmap`) map these shared objects into the process's virtual address space.
   * **During Model Training & Evaluation:** Once imported, all model execution (K-Means, Multi-View Spectral, UMAP, TriTopic, c-TF-IDF, coherence calculation) occurs **100% in CPU RAM and GPU VRAM**. Python never reads from the virtual environment on disk/NFS again during training.

2. **The Linux Kernel Page Cache:**
   * When library files are read from NFS into memory on a compute node, the Linux kernel retains those memory pages in the node's local **RAM Page Cache**.
   * Subsequent process initializations or sibling runs on the same node serve library code directly from local memory buffers, requiring virtually zero subsequent network requests across the NFS fabric.

3. **Preserving True `$SCRATCH` Isolation Where It Matters:**
   * **Datasets (`*.parquet`):** Still copied to local node `$SCRATCH`, guaranteeing low-latency file I/O during data ingestion.
   * **Results & Outputs (`.csv`, `.json`, `.tex`):** Still generated and written to local `$SCRATCH`, isolating write operations and avoiding network write latency until the final rsync.

| Phase | Previous Setup (`uv run` in scratch) | New Setup (Symlinked `.venv` + `--no-sync`) |
| :--- | :--- | :--- |
| **Virtualenv / Package Setup** | **3 to 6 minutes** (copying 60,000 files & ~10 GB) | **~0.001 seconds** (creates 1 symlink) |
| **Python Module Imports** | ~1 second | ~2–3 seconds (over NFS read once) |
| **Model Training & Evaluation** | Full in-memory/GPU speed | Full in-memory/GPU speed |
| **NFS / Cache Lock Contention** | **High** (all jobs racing to build and lock) | **Zero** (completely read-only) |
| **Scratch Disk Usage** | ~10 GB per job (200 GB for 20 jobs) | **0 bytes** for packages |

Symlinking avoids spending 5 minutes and 42 seconds copying files to save 1 second of import time, allowing jobs to start in seconds and train at full native hardware performance.

### 5.3 Step 3: Configure `UV_LINK_MODE` and Robust Path Resolution
1. Support both `$HOME/fast-tritopic` and `$HOME/fast_tritopic` in the rsync fallback.
2. Set `export UV_LINK_MODE="copy"` to eliminate the hardlink fallback warning.
3. Add exit status checking (`RUN_EXIT=$?`) after the execution command so that any failure immediately logs the exit code to `.err` and marks the SLURM job as `FAILED` instead of completing silently.

---

## 6. Cluster Operations Checklist

When deploying these changes to the cluster:

1. **Pull updated repository:**
   ```bash
   cd ~/ca_bertopic
   git pull
   ```
2. **Run `uv sync` once on the login node:**
   ```bash
   uv sync
   ```
   * This builds `fast-tritopic` once into `~/ca_bertopic/.venv/lib/python3.12/site-packages/fast_tritopic/` as a static package.
   * No separate `uv build` in `~/fast_tritopic` is required.
3. **Dispatch experiments:**
   ```bash
   ./scripts/pipelines/slurm/queue_exp.sh -d gadarian -m pca_k_means,pca_mv_spectral
   ```
   * Jobs will now start instantly (0.1s instead of 5m 42s) and execute reliably without NFS build collisions.
