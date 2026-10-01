import pytest

from src.optimizer import generate_hyperparameter_combinations

# A minimal experiment config for testing
MOCK_EXPERIMENT_CONFIG = {"experiment": {}}


def test_config_with_no_search_space():
    """
    Tests that a single config is returned when the config doesn't
    define a hyperparameter search space (i.e., no lists of values).
    """
    model_config = {"id": "test", "clustering": {"params": {"n_clusters": 50}}}
    combinations = generate_hyperparameter_combinations(model_config)

    assert len(combinations) == 1
    assert combinations[0][0] == model_config
    assert combinations[0][1] == {}


def test_single_hyperparameter():
    """
    Tests generation with a single hyperparameter list.
    """
    model_config = {
        "id": "test",
        "clustering": {"params": {"n_clusters": [10, 20, 30]}},
    }
    combinations = generate_hyperparameter_combinations(model_config)

    assert len(combinations) == 3

    # Check generated configs
    assert combinations[0][0]["clustering"]["params"]["n_clusters"] == 10
    assert combinations[1][0]["clustering"]["params"]["n_clusters"] == 20
    assert combinations[2][0]["clustering"]["params"]["n_clusters"] == 30

    # Check varied params dict
    assert combinations[0][1] == {"clustering.params.n_clusters": 10}
    assert combinations[1][1] == {"clustering.params.n_clusters": 20}
    assert combinations[2][1] == {"clustering.params.n_clusters": 30}


def test_multiple_hyperparameters():
    """
    Tests the cartesian product of multiple hyperparameter lists.
    """
    model_config = {
        "id": "test",
        "dimensionality_reduction": {"params": {"n_components": [5, 10]}},
        "clustering": {"params": {"n_clusters": [100, 200]}},
    }
    combinations = generate_hyperparameter_combinations(model_config)

    assert len(combinations) == 4  # 2 * 2

    # Check that all combinations are present
    expected_configs = [(5, 100), (5, 200), (10, 100), (10, 200)]
    generated_configs = [
        (
            c[0]["dimensionality_reduction"]["params"]["n_components"],
            c[0]["clustering"]["params"]["n_clusters"],
        )
        for c in combinations
    ]
    assert sorted(generated_configs) == sorted(expected_configs)

    # Check one of the varied_params dicts
    assert {
        "dimensionality_reduction.params.n_components": 5,
        "clustering.params.n_clusters": 200,
    } in [c[1] for c in combinations]


def test_string_list_is_ignored():
    """
    Tests that a list of strings is correctly ignored and not treated as a
    hyperparameter to be varied.
    """
    model_config = {
        "id": "test",
        "representation_model": {
            "type": "KeyBERT",
            "params": {
                # This is a valid parameter value, not a list to iterate over
                "stop_words": ["english", "custom"]
            },
        },
        "clustering": {"params": {"n_clusters": [10, 20]}},
    }
    combinations = generate_hyperparameter_combinations(model_config)

    # Should only generate combinations for n_clusters
    assert len(combinations) == 2
    assert combinations[0][0]["representation_model"]["params"]["stop_words"] == [
        "english",
        "custom",
    ]
    assert combinations[1][0]["representation_model"]["params"]["stop_words"] == [
        "english",
        "custom",
    ]
    assert combinations[0][0]["clustering"]["params"]["n_clusters"] == 10
    assert combinations[1][0]["clustering"]["params"]["n_clusters"] == 20


def test_range_hyperparameter():
    """
    Tests generation with a range hyperparameter.
    """
    model_config = {
        "id": "test",
        "clustering": {"params": {"n_clusters": {"start": 10, "stop": 31, "step": 10}}},
    }
    combinations = generate_hyperparameter_combinations(model_config)

    assert len(combinations) == 3

    # Check generated configs
    assert combinations[0][0]["clustering"]["params"]["n_clusters"] == 10
    assert combinations[1][0]["clustering"]["params"]["n_clusters"] == 20
    assert combinations[2][0]["clustering"]["params"]["n_clusters"] == 30

    # Check varied params dict
    assert combinations[0][1] == {"clustering.params.n_clusters": 10}
    assert combinations[1][1] == {"clustering.params.n_clusters": 20}
    assert combinations[2][1] == {"clustering.params.n_clusters": 30}


def test_float_range_hyperparameter():
    """
    Tests generation with a float range hyperparameter.
    """
    model_config = {
        "id": "test",
        "dimensionality_reduction": {
            "params": {"some_float": {"start": 0.1, "stop": 0.31, "step": 0.1}}
        },
    }
    combinations = generate_hyperparameter_combinations(model_config)

    assert len(combinations) == 3  # 0.1, 0.2, 0.3

    # Check generated configs and varied params dict
    # Note: due to float precision, it's better to check if they are close
    generated_values = [
        c[0]["dimensionality_reduction"]["params"]["some_float"] for c in combinations
    ]
    expected_values = [0.1, 0.2, 0.3]
    assert all(pytest.approx(g) == e for g, e in zip(generated_values, expected_values))

    varied_params = [c[1] for c in combinations]
    expected_varied = [
        {"dimensionality_reduction.params.some_float": 0.1},
        {"dimensionality_reduction.params.some_float": 0.2},
        {"dimensionality_reduction.params.some_float": 0.3},
    ]
    assert all(
        pytest.approx(v["dimensionality_reduction.params.some_float"])
        == e["dimensionality_reduction.params.some_float"]
        for v, e in zip(varied_params, expected_varied)
    )


def test_mixed_hyperparameters():
    """
    Tests a mix of list and range hyperparameters.
    """
    model_config = {
        "id": "test",
        "dimensionality_reduction": {"params": {"n_components": [5, 10]}},
        "clustering": {
            "params": {"n_clusters": {"start": 100, "stop": 201, "step": 100}}
        },
    }
    combinations = generate_hyperparameter_combinations(model_config)

    assert len(combinations) == 4  # 2 * 2

    expected_configs = [(5, 100), (5, 200), (10, 100), (10, 200)]
    generated_configs = [
        (
            c[0]["dimensionality_reduction"]["params"]["n_components"],
            c[0]["clustering"]["params"]["n_clusters"],
        )
        for c in combinations
    ]
    assert sorted(generated_configs) == sorted(expected_configs)


def test_determinism_via_sorting():
    """
    Tests that the order of hyperparameter combinations is deterministic
    by ensuring it doesn't depend on the dictionary order of the config.
    """
    # Create two configs with the same keys but different insertion order
    config_a = {
        "dimensionality_reduction": {
            "params": {"n_components": [5, 10], "some_param": [1, 2]}
        },
        "clustering": {"params": {"n_clusters": [100, 200]}},
    }

    config_b = {
        "dimensionality_reduction": {
            "params": {"some_param": [1, 2], "n_components": [5, 10]}
        },
        "clustering": {"params": {"n_clusters": [100, 200]}},
    }

    comb_a = generate_hyperparameter_combinations(config_a)
    comb_b = generate_hyperparameter_combinations(config_b)

    params_a = [c[1] for c in comb_a]
    params_b = [c[1] for c in comb_b]

    assert params_a == params_b


def test_bertopic_hyperparameter():
    """
    Tests generation with a bertopic hyperparameter list.
    """
    model_config = {
        "id": "test",
        "bertopic": {"params": {"nr_topics": [10, 20, 30]}},
    }
    combinations = generate_hyperparameter_combinations(model_config)

    assert len(combinations) == 3

    # Check generated configs
    assert combinations[0][0]["bertopic"]["params"]["nr_topics"] == 10
    assert combinations[1][0]["bertopic"]["params"]["nr_topics"] == 20
    assert combinations[2][0]["bertopic"]["params"]["nr_topics"] == 30

    # Check varied params dict
    assert combinations[0][1] == {"bertopic.params.nr_topics": 10}
    assert combinations[1][1] == {"bertopic.params.nr_topics": 20}
    assert combinations[2][1] == {"bertopic.params.nr_topics": 30}


def test_optimizer_run_baseline_with_scaled_metadata():
    """
    Tests that Optimizer.run() successfully trains a standard BERTopic baseline
    model without failing when scaled_metadata is provided.
    """
    import numpy as np

    from src.optimizer import Optimizer

    model_config = {
        "id": "baseline",
        "is_baseline": True,
        "dimensionality_reduction": {
            "type": "umap",
            "params": {"n_neighbors": 3, "min_dist": 0.0, "metric": "cosine"},
        },
        "clustering": {
            "type": "hdbscan",
            "params": {"min_cluster_size": 2},
        },
        "bertopic": {
            "params": {"nr_topics": [2, 3]},
        },
    }

    experiment_config = {
        "experiment": {
            "dataset_path": "data/mock_dataset.parquet",
            "coherence_metrics": [],
            "diversity_metrics": [],
        }
    }

    texts = [
        "apple banana orange fruit salad",
        "pear apple grape fruit smoothie",
        "car truck vehicle engine motor",
        "bus train vehicle diesel motor",
        "python code software programming test",
        "java script software developer coding",
    ]
    embeddings = np.random.rand(len(texts), 10)
    scaled_metadata = np.random.rand(len(texts), 4)

    optimizer = Optimizer(
        texts=texts,
        embeddings=embeddings,
        scaled_metadata=scaled_metadata,
        model_config=model_config,
        experiment_config=experiment_config,
        experiment_id="test_baseline_opt",
        random_state=42,
        file_timestamp="20260101-000000",
    )

    optimizer.run()

    assert len(optimizer.results) == 2
    assert optimizer.results[0]["model_name"] == "baseline_1"
    assert optimizer.results[1]["model_name"] == "baseline_2"
    assert "n_topics" in optimizer.results[0]


def test_tritopic_grid_keeps_realized_topic_count(monkeypatch):
    """A grid parameter named like a metric must not overwrite the metric.

    TriTopic requests topics through `params.n_topics`, which shares its name
    with the realized topic count reported by training.
    """
    import src.models as models
    import src.run_provenance as run_provenance
    import src.training as training
    import src.utils as utils
    from src.optimizer import Optimizer

    monkeypatch.setattr(
        models, "create_topic_model_instance", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        training,
        "train_and_evaluate",
        lambda **kwargs: (
            {"model_name": kwargs["model_id"], "n_topics": 7, "outliers": 0},
            object(),
        ),
    )
    monkeypatch.setattr(run_provenance, "collect_run_provenance", lambda **kwargs: {})
    monkeypatch.setattr(utils, "extract_qualitative_data", lambda *args: None)

    optimizer = Optimizer(
        texts=["a", "b"],
        embeddings=None,
        scaled_metadata=None,
        model_config={
            "id": "tritopic",
            "type": "tritopic",
            "params": {"n_topics": [10, 20]},
        },
        experiment_config={
            "experiment": {"dataset_path": "data/toy_embeddings.parquet"}
        },
        experiment_id="test_tritopic",
        random_state=[1],
        file_timestamp="20261001-000000",
    )
    optimizer.run()

    assert [row["requested_topics"] for row in optimizer.results] == [10, 20]
    assert [row["n_topics"] for row in optimizer.results] == [7, 7]


def test_requested_topic_setting_per_model_family():
    from src.document_assignments import requested_topic_setting

    assert requested_topic_setting({"bertopic": {"params": {"nr_topics": 30}}}) == 30
    assert requested_topic_setting({"clustering": {"params": {"n_clusters": 20}}}) == 20
    assert requested_topic_setting({"params": {"n_topics": 40}}) == 40
    assert requested_topic_setting({"id": "stm"}) is None
