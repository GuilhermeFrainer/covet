import logging
import random
from pathlib import Path

import numpy as np
import polars as pl
import yaml


def extract_stm_qualitative_data(
    theta: np.ndarray,
    beta: np.ndarray,
    vocab: list[str],
    documents: list[str],
    model_id: str,
    metadata: dict,
    topk: int = 10,
    top_words: list[list[str]] | None = None,
) -> pl.DataFrame:
    """
    Extracts qualitative data from STM outputs (theta, beta) in a format
    compatible with the existing BERTopic qualitative data schema.

    Pass the evaluated `top_words` so the saved representations are the words
    that were scored; otherwise the `topk` most probable words are used.
    """
    import json

    n_topics = beta.shape[0]

    # 1. Topic counts (sum of probabilities)
    counts = np.sum(theta, axis=0)

    # 2. Representations (top words)
    if top_words is None:
        top_words = []
        for i in range(n_topics):
            top_indices = np.argsort(beta[i])[::-1][:topk]
            top_words.append([vocab[idx] for idx in top_indices])

    # 3. Representative documents (top 3 documents for each topic)
    rep_docs = []
    for i in range(n_topics):
        top_doc_indices = np.argsort(theta[:, i])[::-1][:3]
        # Ensure we don't go out of bounds if documents is shorter (shouldn't happen)
        actual_indices = [idx for idx in top_doc_indices if idx < len(documents)]
        rep_docs.append([documents[idx] for idx in actual_indices])

    # 4. Create DataFrame
    data = []
    for i in range(n_topics):
        topic_id = i
        data.append(
            {
                "topic_id": topic_id,
                "count": int(counts[i]),
                "name": f"{topic_id}_" + "_".join(top_words[i][:3]),
                "representation": top_words[i],
                "representative_docs": rep_docs[i],
            }
        )

    df = pl.DataFrame(data)

    # Add model_id at the front
    df = df.with_columns(pl.lit(model_id).alias("model_id"))

    # Add metadata columns at the back
    for key, value in metadata.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        df = df.with_columns(pl.lit(value).alias(key))

    # Reorder columns
    topic_cols = ["topic_id", "count", "name", "representation", "representative_docs"]
    metadata_keys = list(metadata.keys())
    final_order = ["model_id"] + topic_cols + metadata_keys

    return df.select(final_order)


def load_config(exp_name: str, experiments_dir: Path) -> dict:
    """
    Resolves path and loads the YAML config, supporting inheritance via 'extends'.
    """
    logger = logging.getLogger("pipeline")

    filename = exp_name if exp_name.endswith(".yaml") else f"{exp_name}.yaml"
    config_path = experiments_dir / filename

    if not config_path.exists():
        matches = [
            m for m in experiments_dir.rglob(filename) if "archive" not in m.parts
        ]
        if matches:
            config_path = matches[0]
        else:
            all_matches = list(experiments_dir.rglob(filename))
            if all_matches:
                config_path = all_matches[0]
            else:
                raise FileNotFoundError(f"Experiment file {config_path} not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "extends" in config:
        base_rel_path = config.pop("extends")
        # Ensure we can load from subdirectories like 'datasets/'
        base_path = experiments_dir / base_rel_path

        if not base_path.exists():
            raise FileNotFoundError(f"Base config file {base_path} not found.")

        with open(base_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
            logger.debug(f"Loaded base config from {base_path}")

        # If base file is flat (no 'experiment' key), treat it as 'experiment' data
        if (
            "experiment" not in base_config
            and "models" not in base_config
            and "model" not in base_config
        ):
            base_config = {"experiment": base_config}

        # Merge logic (Replace strategy)
        for key, value in config.items():
            if key == "experiment" and isinstance(value, dict) and key in base_config:
                # Merge the 'experiment' dictionary keys (shallow merge)
                base_config[key].update(value)
            else:
                # Replace other top-level keys (e.g., 'models', 'model')
                base_config[key] = value

        config = base_config

    logger.debug(f"Loaded config from {config_path}")
    return config


def get_random_state(random_state: str | int | list[str | int]) -> int | list[int]:
    if isinstance(random_state, list):
        return [get_random_state(r) for r in random_state]
    if isinstance(random_state, int):
        return random_state
    elif random_state == "random":
        return random.randint(0, 100_000)
    else:
        raise ValueError(f"Invalid random state: {random_state}")


def extract_qualitative_data(
    topic_model, model_id: str, metadata: dict
) -> pl.DataFrame:
    """
    Extracts c-TF-IDF words and representative documents from a BERTopic model.

    Args:
        topic_model: Fitted BERTopic model.
        model_id: Identifier for the model.
        metadata: Dictionary of metadata/hyperparameters to include.

    Returns:
        Polars DataFrame with topic information and metadata.
    """
    import json

    import polars as pl

    # Get topic info from model (returns a pandas DataFrame)
    topic_info = topic_model.get_topic_info().copy()

    # For TriTopic: prefer All_Keywords for complete representation if present
    if "All_Keywords" in topic_info.columns:
        topic_info["Keywords"] = topic_info["All_Keywords"]

    # For models where Representative_Docs contains integer indices (like TriTopic),
    # map them to the actual document texts if documents_ is available.
    if (
        hasattr(topic_model, "documents_")
        and topic_model.documents_ is not None
        and "Representative_Docs" in topic_info.columns
        and len(topic_info) > 0
    ):
        docs = topic_model.documents_
        first_docs = topic_info["Representative_Docs"].iloc[0]
        if (
            isinstance(first_docs, (list, np.ndarray))
            and len(first_docs) > 0
            and isinstance(first_docs[0], (int, np.integer))
        ):

            def map_docs(row_docs):
                if isinstance(row_docs, (list, np.ndarray)):
                    mapped = [
                        str(docs[idx])
                        for idx in row_docs
                        if isinstance(idx, (int, np.integer)) and 0 <= idx < len(docs)
                    ]
                    if mapped:
                        return mapped
                return row_docs

            topic_info["Representative_Docs"] = topic_info["Representative_Docs"].apply(
                map_docs
            )

    # Convert to Polars
    df = pl.from_pandas(topic_info)

    # Standardize column names to snake_case
    rename_dict = {
        "Topic": "topic_id",
        "Count": "count",
        "Size": "count",
        "Name": "name",
        "Label": "name",
        "Keywords": "representation",
        "Representation": "representation",
        "Representative_Docs": "representative_docs",
    }
    # Only rename if they exist
    actual_rename = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(actual_rename)

    # Add model_id at the front
    df = df.with_columns(pl.lit(model_id).alias("model_id"))

    # Add metadata columns at the back
    for key, value in metadata.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        df = df.with_columns(pl.lit(value).alias(key))

    # Standardize column types for representation and representative_docs
    if "representation" in df.columns:
        dtype = df["representation"].dtype
        if dtype in (pl.String, pl.Utf8):

            def parse_repr(x):
                if x is None:
                    return []
                if isinstance(x, list):
                    return [str(w) for w in x]
                if not isinstance(x, str) or not x.strip():
                    return []
                if x.startswith("[") and x.endswith("]"):
                    try:
                        parsed = json.loads(x)
                        if isinstance(parsed, list):
                            return [str(w) for w in parsed]
                    except Exception:
                        pass
                return [w.strip() for w in x.split(",") if w.strip()]

            df = df.with_columns(
                pl.col("representation").map_elements(
                    parse_repr, return_dtype=pl.List(pl.String)
                )
            )
        elif isinstance(dtype, pl.List):
            df = df.with_columns(pl.col("representation").cast(pl.List(pl.String)))

    if "representative_docs" in df.columns:
        dtype = df["representative_docs"].dtype
        if dtype != pl.List(pl.String):
            if isinstance(dtype, pl.List):
                df = df.with_columns(
                    pl.col("representative_docs").cast(pl.List(pl.String))
                )
            elif dtype in (pl.String, pl.Utf8):

                def parse_docs(x):
                    if x is None:
                        return []
                    if isinstance(x, list):
                        return [str(d) for d in x]
                    if not isinstance(x, str) or not x.strip():
                        return []
                    if x.startswith("[") and x.endswith("]"):
                        try:
                            parsed = json.loads(x)
                            if isinstance(parsed, list):
                                return [str(d) for d in parsed]
                        except Exception:
                            pass
                    return [x]

                df = df.with_columns(
                    pl.col("representative_docs").map_elements(
                        parse_docs, return_dtype=pl.List(pl.String)
                    )
                )
            else:
                df = df.with_columns(
                    pl.col("representative_docs").map_elements(
                        lambda x: [str(x)] if x is not None else [],
                        return_dtype=pl.List(pl.String),
                    )
                )

    # Reorder columns: model_id first, then topic info, then metadata
    topic_cols = ["topic_id", "count", "name", "representation", "representative_docs"]
    existing_topic_cols = [c for c in topic_cols if c in df.columns]
    metadata_keys = list(metadata.keys())

    final_order = ["model_id"] + existing_topic_cols + metadata_keys
    return df.select(final_order)
