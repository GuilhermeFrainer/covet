# Archived documentation

These documents record completed plans, resolved incidents, and finished
investigations. They are kept for historical context only. **Do not follow them
as current instructions.** Commands, file paths, and statuses in them may be
outdated. For current workflows, see the [README](../../README.md) and the
documents directly under [docs/](..).

| Document | What it records | Why archived |
| --- | --- | --- |
| [experiment_integrity_repair_plan.md](experiment_integrity_repair_plan.md) | Plan to repair UMAP dimensionality, config mutation, provenance, and CSV precision (2026-09-16) | Repairs implemented; open items live in [REPOSITORY_ISSUES.md](../REPOSITORY_ISSUES.md) |
| [EXPERIMENT_INTEGRITY_NEXT_STEPS.md](EXPERIMENT_INTEGRITY_NEXT_STEPS.md) | Stage-by-stage execution guide for the repair plan | Stages 1–6 completed |
| [document_assignment_export_plan.md](document_assignment_export_plan.md) | Design plan for per-document assignment exports | Implemented; current behavior is in [document_assignment_exports.md](../document_assignment_exports.md) |
| [fast_tritopic_implementation_plan.md](fast_tritopic_implementation_plan.md) | Plan for the vectorized FastTriTopic implementation | Implemented in the sibling `fast-tritopic` package |
| [tritopic_sparse_efficiency_investigation.md](tritopic_sparse_efficiency_investigation.md) | Profiling of TriTopic's sparse graph construction | Investigation concluded; led to FastTriTopic |
| [slurm_fast_tritopic_concurrency_issue.md](slurm_fast_tritopic_concurrency_issue.md) | Root cause of intermittent SLURM job failures after adding `fast-tritopic` | Fix applied in `scripts/experiments/slurm_job.sh` (shared `.venv` symlink, `--no-sync`, `UV_LINK_MODE=copy`, exit-code propagation) |
