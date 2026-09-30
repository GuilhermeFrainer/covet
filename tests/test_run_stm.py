import pathlib
import sys

import pytest

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_stm import resolve_stm_inputs  # noqa: E402
from src.utils import load_config  # noqa: E402

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_unstemmed_config_uses_unstemmed_inputs():
    name, rds_path, bow_path = resolve_stm_inputs(
        {
            "dataset_path": "data/processed/trump_embeddings.parquet",
            "text_col": "clean_text",
        }
    )
    assert name == "trump"
    assert rds_path == PROCESSED_DIR / "trump_stm_data.rds"
    assert bow_path == PROCESSED_DIR / "trump_bow.parquet"


def test_stemmed_config_uses_stemmed_inputs():
    name, rds_path, bow_path = resolve_stm_inputs(
        {
            "dataset_path": "data/processed/trump_embeddings.parquet",
            "text_col": "clean_text_stemmed",
        }
    )
    assert name == "trump"
    assert rds_path == PROCESSED_DIR / "trump_stemmed_stm_data.rds"
    assert bow_path == PROCESSED_DIR / "trump_stemmed_bow.parquet"


@pytest.mark.parametrize(
    ("exp_name", "expected_prefix"),
    [
        ("trump/trump_standard_stm", "trump"),
        ("trump_stemmed/trump_standard_stm", "trump_stemmed"),
        ("yelp/yelp_standard_stm", "yelp_s10000"),
        ("yelp_stemmed/yelp_standard_stm", "yelp_s10000_stemmed"),
        ("trump_s25000/trump_s25000_standard_stm", "trump_s25000"),
        ("trump_s25000_stemmed/trump_s25000_standard_stm", "trump_s25000_stemmed"),
    ],
)
def test_active_stm_configs_match_build_bow_outputs(exp_name, expected_prefix):
    config = load_config(exp_name, EXPERIMENTS_DIR)
    _, rds_path, bow_path = resolve_stm_inputs(config["experiment"])
    assert rds_path.name == f"{expected_prefix}_stm_data.rds"
    assert bow_path.name == f"{expected_prefix}_bow.parquet"
