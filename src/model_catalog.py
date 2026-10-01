"""Dataset-independent model roles and baseline relationships for presentation."""

import re
from pathlib import Path

import polars as pl
import yaml

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config/model_catalog.yaml"
FIELDS = ("catalog_id", "display_label", "priority", "role", "family", "baseline_id")


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently overwriting catalog entries."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate catalog key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_catalog(path=CATALOG_PATH):
    """Load and validate the explicit catalog, including aliases and references."""
    with Path(path).open(encoding="utf-8") as stream:
        try:
            catalog = yaml.load(stream, Loader=UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid catalog YAML: {exc}") from exc
    if (
        not isinstance(catalog, dict)
        or set(catalog) != {"schema_version", "models"}
        or catalog["schema_version"] != 1
    ):
        raise ValueError("Unsupported model catalog schema")
    models = catalog["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("Catalog models must be a nonempty mapping")
    aliases = set(models)
    required = {"label", "priority", "role", "family", "baseline_id", "reduction", "clustering"}
    optional_text = {"short_label", "latex_label", "change"}
    for model_id, entry in models.items():
        if not isinstance(model_id, str) or not isinstance(entry, dict):
            raise ValueError("Model identifiers and entries must be strings and mappings")
        allowed = required | optional_text | {"aliases"}
        if not required <= entry.keys() or entry.keys() - allowed:
            raise ValueError(f"Invalid fields for {model_id}")
        if any(
            not isinstance(entry[field], str) or not entry[field]
            for field in (required - {"baseline_id"}) | (optional_text & entry.keys())
        ):
            raise ValueError(f"Model fields must be nonempty strings: {model_id}")
        if not isinstance(entry.get("aliases", []), list) or any(
            not isinstance(alias, str) or not alias for alias in entry.get("aliases", [])
        ):
            raise ValueError(f"Aliases must be a list of strings: {model_id}")
        if entry["priority"] not in {"primary", "secondary"}:
            raise ValueError(f"Invalid priority for {model_id}")
        if entry["role"] not in {"baseline", "ablation", "external_baseline"}:
            raise ValueError(f"Invalid role for {model_id}")
        if entry["family"] not in {"hdbscan", "spectral", "k_means", "external"}:
            raise ValueError(f"Invalid family for {model_id}")
        baseline = entry["baseline_id"]
        if entry["role"] == "ablation":
            if (
                not isinstance(baseline, str)
                or not isinstance(models.get(baseline), dict)
                or models[baseline].get("role") != "baseline"
            ):
                raise ValueError(f"Invalid baseline reference for {model_id}: {baseline}")
            if models[baseline].get("family") != entry["family"]:
                raise ValueError(f"Baseline family mismatch for {model_id}")
        elif baseline is not None:
            raise ValueError(f"Only ablations may reference a baseline: {model_id}")
        for alias in entry.get("aliases", []):
            if alias in aliases:
                raise ValueError(f"Duplicate model alias: {alias}")
            aliases.add(alias)
    return models


def resolve_model_id(name, models):
    """Normalize historical run suffixes without merging info0 or geometry variants."""
    if not isinstance(name, str):
        return None
    name = re.sub(r"^stemmed_", "", name)
    name = re.sub(r"_seed.*$", "", name)
    name = re.sub(r"_\d+$", "", name)
    if re.fullmatch(r"stm_k\d+", name):
        name = "stm"
    if name in models:
        return name
    for model_id, entry in models.items():
        if name in entry.get("aliases", []):
            return model_id
    return None


def annotate_models(df, models, identity_column=None):
    """Add catalog columns while preserving unknown rows and original identifiers."""
    columns = [identity_column] if identity_column else ["model_type", "model_name", "model_id"]
    columns = [column for column in columns if column in df.columns]
    records = []
    for row in df.select(columns).iter_rows(named=True) if columns else [{}] * df.height:
        model_id = next((resolved for value in row.values()
                         if (resolved := resolve_model_id(value, models))), None)
        entry = models.get(model_id, {})
        records.append({
            "catalog_id": model_id or "unclassified",
            "display_label": entry.get("label", "Unclassified"),
            "priority": entry.get("priority", "unclassified"),
            "role": entry.get("role", "unclassified"),
            "family": entry.get("family", "unclassified"),
            "baseline_id": entry.get("baseline_id"),
        })
    annotations = pl.DataFrame(records, schema={field: pl.String for field in FIELDS})
    return df.with_columns(annotations.get_columns())


def filter_catalog(df, priority="primary", families=(), roles=(), baselines=(), include_external=False):
    """Select a catalog scope; baseline selection includes the reference itself."""
    mask = pl.lit(True)
    if priority != "all":
        mask &= pl.col("priority") == priority
    relationship = pl.lit(True)
    if families:
        relationship &= pl.col("family").is_in(families)
    if baselines:
        relationship &= (pl.col("baseline_id").is_in(baselines).fill_null(False)
                         | pl.col("catalog_id").is_in(baselines))
    if include_external:
        relationship |= pl.col("role") == "external_baseline"
    mask &= relationship
    if roles:
        mask &= pl.col("role").is_in(roles)
    return df.filter(mask)


def kmeans_algorithms(df):
    """Return the actual K-means spellings present in results for optional exclusions."""
    if "clustering_algo" not in df.columns:
        return ()
    return tuple(value for value in df["clustering_algo"].drop_nulls().unique().sort()
                 if "kmeans" in re.sub(r"[^a-z]", "", value.lower()))


def configuration_models(experiments):
    """Read model identities from configs, including configs with no results yet."""
    identities = {}
    for experiment in experiments:
        with Path(experiment["file_path"]).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        model_id = config.get("model", {}).get("id")
        if model_id is None and experiment["canonical_name"].endswith("_stm"):
            model_id = "stm"
        identities[(experiment["dataset_label"], experiment["canonical_name"])] = model_id
    return identities


def sort_catalog(df, identity="catalog_id"):
    """Keep families together and show their reference baseline first."""
    return df.sort(["family", pl.col("role") != "baseline", identity])


def annotate_coverage(matrix, experiments, models):
    """Classify coverage by config identity, not by whether results exist."""
    identities = configuration_models(experiments)
    matrix = matrix.with_columns(pl.Series("model_id", [
        identities.get((row["dataset_label"], row["experiment_name"]))
        for row in matrix.iter_rows(named=True)
    ], dtype=pl.String))
    return annotate_models(matrix, models)
