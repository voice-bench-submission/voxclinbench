"""Unit tests for voxbench.eval.delong_p and paired_bootstrap_p.

Covers:
    (a) identical predictors -> p ~ 1.0
    (b) strongly different predictors -> p < 0.05
    (c) edge cases: single-class labels and NaN inputs raise ValueError
    (d) sanity check: DeLong AUCs match sklearn roc_auc_score
    (e) DeLong vs paired-bootstrap agree in sign and rough magnitude
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from voxbench.eval import delong_p, paired_bootstrap_p


def _make_scores(n: int = 200, seed: int = 0,
                 noise_a: float = 0.5, noise_b: float = 0.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    signal = y.astype(float)
    a = signal + rng.normal(0, noise_a, size=n)
    b = signal + rng.normal(0, noise_b, size=n)
    return y, a, b


def test_delong_identical_predictors_pvalue_high():
    y, a, _ = _make_scores(n=300, seed=1)
    diff, p = delong_p(y, a, a)
    assert diff == 0.0
    assert p >= 0.95


def test_delong_strongly_different_predictors_pvalue_low():
    rng = np.random.default_rng(42)
    n = 400
    y = rng.integers(0, 2, size=n)
    # a is near-perfect, b is near-random
    a = y + rng.normal(0, 0.1, size=n)
    b = rng.normal(0, 1.0, size=n)
    diff, p = delong_p(y, a, b)
    assert diff > 0.3
    assert p < 0.05


def test_delong_auc_matches_sklearn():
    y, a, b = _make_scores(n=250, seed=7, noise_a=0.4, noise_b=1.2)
    diff, _ = delong_p(y, a, b)
    expected = roc_auc_score(y, a) - roc_auc_score(y, b)
    assert abs(diff - expected) < 1e-9


def test_delong_rejects_single_class_labels():
    y = np.zeros(50, dtype=int)
    a = np.random.default_rng(0).normal(size=50)
    b = np.random.default_rng(1).normal(size=50)
    with pytest.raises(ValueError):
        delong_p(y, a, b)


def test_delong_rejects_nan_inputs():
    y, a, b = _make_scores(n=100, seed=2)
    a = a.copy()
    a[0] = np.nan
    with pytest.raises(ValueError):
        delong_p(y, a, b)


def test_delong_rejects_shape_mismatch():
    y, a, _ = _make_scores(n=100, seed=3)
    b = np.zeros(99)
    with pytest.raises(ValueError):
        delong_p(y, a, b)


def test_delong_agrees_with_paired_bootstrap_sign():
    y, a, b = _make_scores(n=300, seed=11, noise_a=0.3, noise_b=1.0)
    d_diff, d_p = delong_p(y, a, b)
    bs_diff, bs_p = paired_bootstrap_p(y, a, b, resamples=500, seed=11)
    assert np.sign(d_diff) == np.sign(bs_diff)
    assert abs(d_diff - bs_diff) < 1e-9
    # Both should call it significant given large separation
    assert d_p < 0.05
    assert bs_p < 0.1
