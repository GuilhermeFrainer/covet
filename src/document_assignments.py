"""Source-aligned, per-execution artifacts independent of result archival."""

import hashlib
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl

from src import metadata_alignment, run_provenance, utils


def file_checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def estimator_settings(model):
    """Capture effective public estimator parameters without fitted arrays."""

    def describe(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (np.ndarray, pl.DataFrame)):
            return {"class": type(value).__name__, "shape": list(value.shape)}
        if isinstance(value, dict):
            return {k: describe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [describe(v) for v in value]
        return {"class": f"{type(value).__module__}.{type(value).__name__}"}

    settings = {}
    for name in ("umap_model", "hdbscan_model", "vectorizer_model"):
        estimator = getattr(model, name, None)
        if estimator is None:
            continue
        settings[name] = {"class": type(estimator).__name__}
        if hasattr(estimator, "get_params"):
            settings[name]["params"] = describe(estimator.get_params(deep=False))
        inner = getattr(estimator, "model", None)
        if inner is not None and hasattr(inner, "get_params"):
            settings[name]["wrapped_params"] = describe(inner.get_params(deep=False))
    settings["nr_topics"] = getattr(model, "nr_topics", None)
    settings["calculate_probabilities"] = getattr(
        model, "calculate_probabilities", None
    )
    return settings


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_assignments(model, documents, topic_table, *, is_tritopic=False):
    """Use the public final label API, then check it against exported topic counts."""
    labels = np.asarray(model.labels_ if is_tritopic else model.topics_)
    if labels.ndim != 1 or len(labels) != len(documents):
        raise ValueError("Final labels must be one-dimensional and input-aligned")
    if labels.dtype.kind not in "iu":
        raise ValueError("Final topic IDs must be non-null integers")
    if documents["input_position"].to_list() != list(range(len(documents))):
        raise ValueError("Input positions must preserve contiguous training order")
    keys = documents["source_document_key"]
    if keys.null_count() or keys.n_unique() != len(documents):
        raise ValueError("Source document keys must be unique and non-null")
    counts = Counter(int(label) for label in labels)
    if topic_table["topic_id"].n_unique() != len(topic_table):
        raise ValueError("Duplicate topic IDs in topic table")
    table_counts = dict(topic_table.select("topic_id", "count").iter_rows())
    if any(table_counts.get(topic) != count for topic, count in counts.items()):
        raise ValueError("Final assignments disagree with exported topic counts")
    if any(counts.get(topic, 0) != count for topic, count in table_counts.items()):
        raise ValueError("Topic table contains inconsistent counts")
    return documents.with_columns(pl.Series("topic_id", labels))


def representative_identities(model, assignments, texts):
    """Resolve indices or topic-scoped text matches; keep duplicates ambiguous."""
    lookup = {}
    for position, (topic, text) in enumerate(zip(assignments["topic_id"], texts)):
        lookup.setdefault((topic, text), []).append(position)
    rows = []
    for row in model.get_topic_info().to_dict("records"):
        topic = int(row["Topic"])
        representatives = row.get("Representative_Docs")
        if representatives is None:
            continue
        for rank, representative in enumerate(representatives, 1):
            if isinstance(representative, (int, np.integer)):
                position = int(representative)
                candidates = (
                    [position]
                    if 0 <= position < len(texts)
                    and assignments["topic_id"][position] == topic
                    else []
                )
                text = texts[position] if candidates else None
            else:
                text = str(representative)
                candidates = lookup.get((topic, text), [])
            rows.append(
                {
                    "topic_id": topic,
                    "representative_rank": rank,
                    "text": text,
                    "resolution": "unique"
                    if len(candidates) == 1
                    else ("ambiguous" if candidates else "unresolved"),
                    "input_position": candidates[0] if len(candidates) == 1 else None,
                    "source_document_key": assignments["source_document_key"][
                        candidates[0]
                    ]
                    if len(candidates) == 1
                    else None,
                    "candidate_source_keys": [
                        assignments["source_document_key"][p] for p in candidates
                    ],
                }
            )
    return rows


class AssignmentRun:
    """Incremental state journal; a manifest commits only complete artifact files."""

    def __init__(
        self, root, dataset, prepared, model_config, metadata, config, enabled=True
    ):
        self.uid = uuid4().hex
        self.directory = Path(root) / dataset / self.uid
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "manifest.json"
        self.prepared = prepared
        self.model_config = model_config
        self.enabled = enabled
        self.alignment = None
        self.metadata = {**metadata, **self.links}
        self.payload = {
            "schema_version": 1,
            "run_uid": self.uid,
            **metadata,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_config": model_config,
            "resolved_config": config,
            "input": prepared.provenance,
            "metadata_condition": "observed",
            "requested_topic_setting": model_config.get("bertopic", {})
            .get("params", {})
            .get(
                "nr_topics",
                model_config.get("clustering", {})
                .get("params", {})
                .get(
                    "n_clusters",
                    model_config.get("params", {}).get(
                        "n_topics", model_config.get("params", {}).get("n_clusters")
                    ),
                ),
            ),
            "assignment_semantics": "hard_cluster",
            "strength_kind": None,
            "fit_status": "pending",
            "evaluation_status": "pending",
            "export_status": "pending" if enabled else "disabled",
            "status": "running",
            "artifacts": {},
        }
        self.save()

    @property
    def links(self):
        # Relative links are portable with the repository/output tree.
        try:
            directory = self.directory.relative_to(run_provenance.PROJECT_ROOT)
        except ValueError:
            directory = self.directory
        return {
            "run_uid": self.uid,
            "assignment_manifest_path": (directory / "manifest.json").as_posix(),
            "assignments_path": (directory / "assignments.parquet").as_posix()
            if self.enabled
            else None,
        }

    def save(self):
        atomic_json(self.path, self.payload)

    def record(self, name, rows=None):
        entry = {"path": name, "sha256": file_checksum(self.directory / name)}
        if rows is not None:
            entry["row_count"] = rows
        self.payload["artifacts"][name] = entry

    def after_fit(self, model, *, is_tritopic=False):
        self.payload["fit_status"] = "success"
        self.payload["provenance"] = run_provenance.collect_run_provenance(
            model,
            self.model_config,
            self.metadata["model_id"],
            run_manifest_path=self.links["assignment_manifest_path"],
        )
        self.payload["effective_estimators"] = estimator_settings(model)
        dependencies = {}
        for package in (
            "bertopic",
            "numpy",
            "polars",
            "scikit-learn",
            "umap-learn",
            "hdbscan",
            "mvlearn",
            "tritopic",
            "fast-tritopic",
        ):
            try:
                dependencies[package] = version(package)
            except PackageNotFoundError:
                dependencies[package] = "not installed"
        self.payload["dependency_versions"] = dependencies
        self.alignment = self._metadata_alignment(model, is_tritopic)
        if self.alignment is not None:
            self.payload["metadata_alignment"] = {
                **self.alignment,
                "method": metadata_alignment.METHOD,
            }
        self.save()
        if not self.enabled:
            return
        try:
            topics = utils.extract_qualitative_data(
                model,
                self.metadata["model_id"],
                {k: v for k, v in self.metadata.items() if k != "model_id"},
            )
            assignments = validate_assignments(
                model, self.prepared.documents, topics, is_tritopic=is_tritopic
            ).with_columns(pl.lit(self.uid).alias("run_uid"))
            temporary = self.directory / ".assignments.parquet.tmp"
            assignments.write_parquet(temporary)
            os.replace(temporary, self.directory / "assignments.parquet")
            self.record("assignments.parquet", len(assignments))
            atomic_json(self.directory / "topics.json", json.loads(topics.write_json()))
            self.record("topics.json", len(topics))
            self.payload.update(
                {
                    "document_count": len(assignments),
                    "noise_count": assignments.filter(pl.col("topic_id") == -1).height,
                    "actual_non_noise_topic_count": assignments.filter(
                        pl.col("topic_id") != -1
                    )["topic_id"].n_unique(),
                    "export_status": "success",
                }
            )
        except Exception as error:
            self.payload.update(export_status="failure", export_error=repr(error))
            self.save()
            raise
        # Identity of representatives is optional; never lose complete assignments.
        try:
            representatives = representative_identities(
                model, assignments, self.prepared.text
            )
            atomic_json(
                self.directory / "representative_documents.json", representatives
            )
            self.record("representative_documents.json", len(representatives))
            self.payload["representative_export_status"] = "success"
        except Exception as error:
            self.payload["representative_export_status"] = "failure"
            self.payload["representative_export_error"] = repr(error)
            logging.getLogger("pipeline").warning(
                "Representative identity export failed: %s", error
            )
        self.save()

    def _metadata_alignment(self, model, is_tritopic):
        """Topic–covariate AMI; a failure here never fails the training run."""
        covariates = getattr(self.prepared, "covariates", None)
        if covariates is None:
            return None
        try:
            labels = model.labels_ if is_tritopic else model.topics_
            return metadata_alignment.metadata_alignment(labels, covariates)
        except Exception as error:
            logging.getLogger("pipeline").warning(
                "Metadata alignment failed for %s: %s", self.metadata["model_id"], error
            )
            self.payload["metadata_alignment_error"] = repr(error)
            return None

    def execute(self, **training_kwargs):
        from src.training import train_and_evaluate

        try:
            metrics, model = train_and_evaluate(
                **training_kwargs, after_fit=self.after_fit
            )
            self.payload["evaluation_status"] = "success"
            metrics.update(self.metadata)
            metrics.update(self.payload.get("provenance", {}))
            metrics.update(self.links)
            by_covariate = {}
            if self.alignment is not None:
                metrics["meta_ami_mean"] = self.alignment["meta_ami_mean"]
                by_covariate = self.alignment["meta_ami_by_covariate"]
            # Per-covariate values vary by dataset, so they stay out of the CSV row.
            atomic_json(
                self.directory / "metrics.json",
                {**metrics, "meta_ami_by_covariate": by_covariate},
            )
            self.record("metrics.json", 1)
            self.payload["status"] = "success"
            self.save()
            return metrics, model
        except BaseException as error:
            if self.payload["fit_status"] == "pending":
                self.payload["fit_status"] = "failure"
            elif self.payload["export_status"] not in ("success", "disabled"):
                self.payload["export_status"] = "failure"
            else:
                if self.payload["evaluation_status"] == "pending":
                    self.payload["evaluation_status"] = "failure"
                else:
                    self.payload["persistence_status"] = "failure"
            self.payload.update(status="failure", error=repr(error))
            if "provenance" in self.payload:
                self.payload["provenance"]["run_status"] = "failure"
            self.save()
            raise


def inspect_run(manifest_path, source_path=None, indices=(3367, 5129)):
    """Validate a snapshot join and summarize raw covariates for complete topics."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["input"]
    source_path = source_path or provenance["dataset_path"]
    if file_checksum(source_path) != provenance["dataset_sha256"]:
        raise ValueError("Source checksum differs from the fitted snapshot")
    for artifact in manifest["artifacts"].values():
        if file_checksum(manifest_path.parent / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {artifact['path']}")
    assignments = pl.read_parquet(manifest_path.parent / "assignments.parquet")
    keys = assignments["source_document_key"].to_list()
    if (
        hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
        != provenance["ordered_keys_sha256"]
    ):
        raise ValueError("Ordered input identity checksum mismatch")
    expected_keys = [
        f"{provenance['dataset_sha256']}:{r}" for r in assignments["source_row_ordinal"]
    ]
    if keys != expected_keys or len(set(keys)) != len(keys):
        raise ValueError("Invalid source identity mapping")
    if assignments["input_position"].to_list() != list(range(len(assignments))):
        raise ValueError("Invalid input order")
    source = pl.read_parquet(source_path).with_row_index("source_row_ordinal")
    joined = assignments.select(
        "input_position", "source_row_ordinal", "topic_id"
    ).join(source, on="source_row_ordinal", how="left", validate="1:1")
    if source.height != provenance["source_row_count"] or any(
        r >= source.height for r in assignments["source_row_ordinal"]
    ):
        raise ValueError("Invalid source row mapping")
    selected = joined.filter(
        pl.col("index" if "index" in joined.columns else "source_row_ordinal").is_in(
            indices
        )
    )
    summaries = {}
    for topic in selected["topic_id"].unique().to_list():
        members = joined.filter(pl.col("topic_id") == topic)
        summary = {"document_count": members.height, "raw_metadata": {}}
        for column in provenance["raw_covariate_types"]:
            values = members[column]
            summary["raw_metadata"][column] = (
                {
                    "min": values.min(),
                    "max": values.max(),
                    "mean": values.mean(),
                    "null_count": values.null_count(),
                }
                if values.dtype.is_numeric()
                else values.value_counts().to_dicts()
            )
        summaries[str(topic)] = summary
    identity_columns = [
        c
        for c in ("index", "id", "source_row_ordinal", "input_position", "topic_id")
        if c in selected.columns
    ]
    return {
        "run_uid": manifest["run_uid"],
        "selected_documents": selected.select(identity_columns).to_dicts(),
        "topics": summaries,
    }
