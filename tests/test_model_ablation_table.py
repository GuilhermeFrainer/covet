import re

import pytest
import yaml

from scripts.analysis.make_model_table import main
from src.make_table import (
    MODEL_LATEX_MAP,
    generate_model_ablation_latex_table,
    generate_model_ablation_rows,
    latex_escape,
)
from src.model_catalog import CATALOG_PATH, load_catalog


@pytest.fixture
def catalog():
    return load_catalog()


def test_default_rows_are_primary_bertopic_variants(catalog):
    rows = generate_model_ablation_rows(catalog)
    ids = [row["model_id"] for row in rows]
    expected = {
        mid
        for mid, entry in catalog.items()
        if entry["priority"] == "primary"
        and entry["role"] != "external_baseline"
        and not mid.startswith("append_umap_w")
    }
    assert set(ids) == expected
    assert len(ids) == len(expected)
    assert not {"stm", "tritopic", "fast_tritopic", "k_means", "pca_hdbscan"} & set(ids)


def test_default_rows_use_curated_text(catalog):
    for row in generate_model_ablation_rows(catalog):
        entry = catalog[row["model_id"]]
        assert row["model"] == entry["short_label"]
        assert row["label"] == entry["latex_label"]
        assert row["change"] == entry["change"]


def test_latex_labels_match_results_tables(catalog):
    for model_id, label in MODEL_LATEX_MAP.items():
        if "latex_label" in catalog.get(model_id, {}):
            assert catalog[model_id]["latex_label"] == label
    rows = generate_model_ablation_rows(catalog, include_weighted_append=True)
    labels = [row["label"] for row in rows]
    assert len(labels) == len(set(labels))


def test_default_system_labels_use_technique_subscripts(catalog):
    labels = [row["label"] for row in generate_model_ablation_rows(catalog)]
    codes = [re.search(r"\\systemshort\}_\\text\{([^}]+)\}", label) for label in labels]
    system_codes = [code.group(1) for code in codes if code]
    assert len(system_codes) == len(set(system_codes))
    assert not re.search(r"\\systemshort\}_\d", " ".join(labels))
    assert catalog["append_umap"]["latex_label"] == r"$\text{\systemshort}_\text{Ap}$"
    assert catalog["mv_hdbscan"]["latex_label"] == r"$\text{\systemshort}_\text{MH}$"
    assert catalog["feature_stacking_hdbscan"]["latex_label"] == (
        r"$\text{\systemshort}_\text{FS}$"
    )


def test_weighted_append_rows_are_optional_and_contiguous(catalog):
    default = [row["model_id"] for row in generate_model_ablation_rows(catalog)]
    assert not any(mid.startswith("append_umap_w") for mid in default)
    assert "append_umap" in default
    rows = generate_model_ablation_rows(catalog, include_weighted_append=True)
    ids = [row["model_id"] for row in rows]
    positions = [i for i, mid in enumerate(ids) if mid.startswith("append_umap_w")]
    assert len(positions) == 6
    assert positions == list(range(positions[0], positions[0] + 6))
    assert ids[positions[0]] == "append_umap_w000"


def test_baseline_precedes_its_ablations(catalog):
    rows = generate_model_ablation_rows(catalog, include_secondary=True)
    seen = set()
    for row in rows:
        entry = catalog[row["model_id"]]
        if entry["role"] == "ablation":
            assert entry["baseline_id"] in seen
            assert row["ablates_on"] == catalog[entry["baseline_id"]]["latex_label"]
        else:
            assert row["ablates_on"] is None
        seen.add(row["model_id"])
    families = [row["family"] for row in rows]
    assert families == sorted(families, key=["hdbscan", "spectral", "k_means"].index)


def test_flags_add_secondary_and_external_models(catalog):
    default = {row["model_id"] for row in generate_model_ablation_rows(catalog)}
    secondary = {
        row["model_id"]
        for row in generate_model_ablation_rows(catalog, include_secondary=True)
    }
    external = generate_model_ablation_rows(catalog, include_external=True)
    assert {
        "k_means",
        "pca_mv_spectral",
        "append_umap_mv_k_means",
    } <= secondary - default
    assert [row["model_id"] for row in external[-3:]] == [
        "fast_tritopic",
        "stm",
        "tritopic",
    ]
    assert all(row["ablates_on"] is None for row in external[-3:])


def test_secondary_fallback_describes_component_changes(catalog):
    rows = generate_model_ablation_rows(catalog, include_secondary=True)
    change = next(row["change"] for row in rows if row["model_id"] == "pca_mv_spectral")
    assert change == (
        "Changes reduction (UMAP to PCA) and clustering "
        "(spectral clustering to multi-view spectral clustering)."
    )


def _table_rows(latex):
    lines = latex.splitlines()
    body = lines[lines.index(r"    \midrule") + 1 : lines.index(r"    \bottomrule")]
    assert body.count(r"    \midrule") == 2
    return [line for line in body if line.endswith(r"\tabularnewline")]


def test_default_latex_fits_one_column(catalog):
    latex = generate_model_ablation_latex_table(catalog)
    rows = _table_rows(latex)
    assert latex.startswith(r"\begin{table}[htbp]")
    assert latex.endswith(r"\end{table}")
    assert r"    Label & Model & Ablates on \\" in latex
    assert "What changes" not in latex
    assert len(rows) == len(generate_model_ablation_rows(catalog))
    assert all(row.count("&") == 2 for row in rows)
    baseline_row = (
        r"    $\text{BERTopic}_\text{H}$ & \raggedright UMAP + HDBSCAN & --- "
        r"\tabularnewline"
    )
    assert baseline_row in latex
    assert r"MV Spectral (info\_view = 0)" in latex
    assert latex_escape(r"a_b & 5% {x}") == r"a\_b \& 5\% \{x\}"


def test_changes_column_spans_both_columns(catalog):
    latex = generate_model_ablation_latex_table(catalog, include_changes=True)
    rows = _table_rows(latex)
    assert latex.startswith(r"\begin{table*}[htbp]")
    assert latex.endswith(r"\end{table*}")
    assert r"    Label & Model & Ablates on & What changes \\" in latex
    assert all(row.count("&") == 3 for row in rows)
    assert r"\raggedright Text-only reference; no metadata. \tabularnewline" in latex


@pytest.mark.parametrize("value", ["", 3])
def test_invalid_optional_text_rejected(tmp_path, value):
    data = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    data["models"]["mv_spectral"]["change"] = value
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_catalog(path)


def test_cli_writes_latex(tmp_path, catalog):
    output = tmp_path / "nested" / "model_ablations.tex"
    main(
        [
            "--latex",
            str(output),
            "--include-external",
            "--include-weighted-append",
            "--include-changes",
        ]
    )
    expected = generate_model_ablation_latex_table(
        catalog,
        include_external=True,
        include_weighted_append=True,
        include_changes=True,
    )
    assert output.read_text(encoding="utf-8") == expected + "\n"


def test_explicit_model_ids_select_and_order_rows(catalog):
    ids = ["baseline", "mv_hdbscan", "append_umap_w010", "umap_spectral", "stm"]
    rows = generate_model_ablation_rows(catalog, model_ids=ids)
    assert [row["model_id"] for row in rows] == ids
    latex = generate_model_ablation_latex_table(catalog, model_ids=ids)
    assert "STM is an external baseline." in latex
    assert "PCA" not in latex and r"\systemshort}_\text{CT" not in latex
