"""Feature-mask and constraint tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adv_ids.attacks.constraints import apply_feature_constraints, project_linf
from adv_ids.data.masks import feature_mask
from adv_ids.data.schemas import CICIDS2017_FEATURES, get_spec
from adv_ids.data.synthetic import generate_synthetic_cicids2017
from adv_ids.data.preprocess import prepare_from_frame


def test_cicids_port_and_flags_frozen():
    mask = feature_mask(CICIDS2017_FEATURES, "cicids2017")
    port = CICIDS2017_FEATURES.index("Destination Port")
    syn = CICIDS2017_FEATURES.index("SYN Flag Count")
    flow = CICIDS2017_FEATURES.index("Flow Duration")
    assert mask[port] == 0.0
    assert mask[syn] == 0.0
    assert mask[flow] == 1.0


def test_constraints_keep_frozen_features():
    rng = np.random.default_rng(0)
    x = rng.random((8, 6)).astype(np.float32)
    x_adv = np.clip(x + 0.5, 0, 1)
    mask = np.array([0, 1, 1, 0, 1, 0], dtype=np.float32)
    out = apply_feature_constraints(x, x_adv, mask)
    np.testing.assert_allclose(out[:, mask < 0.5], x[:, mask < 0.5], atol=1e-6)
    assert out.max() <= 1.0 + 1e-6
    assert out.min() >= -1e-6


def test_linf_projection_respects_eps():
    x = np.full((4, 5), 0.5, dtype=np.float32)
    x_adv = np.full((4, 5), 0.9, dtype=np.float32)
    mask = np.ones(5, dtype=np.float32)
    out = project_linf(x, x_adv, eps=0.1, mask=mask)
    np.testing.assert_allclose(out, 0.6, atol=1e-5)


def test_prepared_mask_matches_schema():
    df = generate_synthetic_cicids2017(n_benign=30, n_attack=20, seed=6)
    bundle = prepare_from_frame(df, val_size=0.0)
    spec = get_spec("cicids2017")
    for i, name in enumerate(bundle.feature_names):
        expected = 0.0 if name in spec.frozen else 1.0
        assert bundle.modifiable_mask[i] == expected
