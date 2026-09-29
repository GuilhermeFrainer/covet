import numpy as np

from src.append_umap import AppendUMAP


def test_fit_transform_appends_metadata_once():
    rng = np.random.default_rng(0)
    text = rng.normal(size=(40, 8))
    meta = rng.uniform(size=(40, 3))
    reducer = AppendUMAP(metadata=meta, n_components=2, random_state=0)
    embedding = reducer.fit_transform(text)
    assert reducer._raw_data.shape[1] == text.shape[1] + meta.shape[1]
    assert embedding.shape == (text.shape[0], 2)
