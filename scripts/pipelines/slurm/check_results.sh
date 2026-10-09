#!/usr/bin/env bash
# Quick health check of the cluster's results in ~/slurm, read-only.
#
# Shows the running jobs, unique seed x k runs per experiment in
# ~/slurm/results, and every crash in the pipeline logs. run_optimizer.py
# logs crashes and still exits 0, so SLURM reports them as COMPLETED; the
# logs are the only place they show up.
#
# From Windows, without copying it to the cluster:
#   wsl -- bash -lc "ssh -i ~/.ssh/pcad_ufrgs gdsfrainer@gppd-hpc.inf.ufrgs.br bash -s" < scripts/pipelines/slurm/check_results.sh
# Logs older than SINCE are ignored (default: the last 24 hours):
#   ... bash -s -- 2026-10-09T10:00 < scripts/pipelines/slurm/check_results.sh

SINCE="${1:-$(date -d '24 hours ago' '+%Y-%m-%dT%H:%M')}"
cd "$HOME/slurm" || exit 1

echo "== Jobs in SLURM"
squeue -u "$(whoami)" -o "%.10i %.40j %.8T %.10M %.12l"

echo
echo "== Unique seed x k runs per experiment (expect 15; STM 5)"
python3 - <<'PY'
import collections
import csv
import glob

runs = collections.defaultdict(set)
docs = collections.defaultdict(set)
for path in glob.glob("results/*.csv"):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            k = next((row[c] for c in ("k", "requested_topics", "nr_topics", "n_clusters", "n_topics")
                      if row.get(c)), "")
            exp = row["experiment_id"]
            runs[exp].add((row.get("seed") or row.get("random_state"), k))
            docs[exp].add(row.get("n_observations"))
for exp in sorted(runs):
    print(f"{len(runs[exp]):3d}  {exp}  (docs: {', '.join(sorted(d for d in docs[exp] if d))})")
PY

echo
echo "== Crashes in pipeline logs since $SINCE"
find logs -maxdepth 1 -name "*.log" -newermt "${SINCE/T/ }" -print0 |
    xargs -0 -r grep -H "Pipeline crashed" | sed -E 's#^logs/##' | cut -c1-220

echo
echo "== Jobs that did not complete since $SINCE"
sacct -S "$SINCE" -X -n --format=JobID,JobName%40,State,Elapsed |
    grep -v -E "COMPLETED|RUNNING|PENDING" || echo "(none)"
