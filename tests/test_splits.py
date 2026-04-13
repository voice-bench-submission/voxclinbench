"""Tests for voxbench.data.make_splits."""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from voxbench.data import make_splits


def _toy_subjects(n: int) -> list[str]:
    return [f"sub-{i:04d}" for i in range(n)]


def test_unstratified_uniform_split() -> None:
    s = make_splits("toy.task", seed=0, subjects=_toy_subjects(100))
    assert len(s.train_subjects) + len(s.val_subjects) + len(s.test_subjects) == 100
    assert len(set(s.train_subjects) & set(s.test_subjects)) == 0
    assert len(set(s.val_subjects) & set(s.test_subjects)) == 0
    # default test_frac 0.20 → 20 test; default val_frac_within_train 0.20 → 16 val; 64 train
    assert len(s.test_subjects) == 20
    assert len(s.val_subjects) == 16
    assert len(s.train_subjects) == 64


def test_determinism_given_seed_and_subjects() -> None:
    subjects = _toy_subjects(200)
    a = make_splits("toy.task", seed=42, subjects=subjects)
    b = make_splits("toy.task", seed=42, subjects=subjects)
    assert a.test_subjects == b.test_subjects
    assert a.train_subjects == b.train_subjects
    assert a.val_subjects == b.val_subjects


def test_input_order_invariance() -> None:
    subjects = _toy_subjects(200)
    reverse = list(reversed(subjects))
    a = make_splits("toy.task", seed=1, subjects=subjects)
    b = make_splits("toy.task", seed=1, subjects=reverse)
    assert a.test_subjects == b.test_subjects


def test_stratified_preserves_class_balance() -> None:
    subjects = _toy_subjects(100)
    labels = [0] * 70 + [1] * 30  # 70/30
    s = make_splits("toy.task", seed=3, subjects=subjects, labels=labels)
    # test set should roughly preserve 70/30 ratio
    test_labels = [labels[int(sid.split("-")[1])] for sid in s.test_subjects]
    pos_frac = sum(test_labels) / len(test_labels)
    assert 0.20 <= pos_frac <= 0.40, f"test positive fraction {pos_frac} outside stratified tolerance"


def test_raises_on_empty_subjects() -> None:
    import pytest as _pt
    with _pt.raises(ValueError):
        make_splits("toy.task", seed=0, subjects=[])


def test_manifest_round_trip() -> None:
    s = make_splits("toy.task", seed=0, subjects=_toy_subjects(50))
    m = s.as_manifest()
    assert set(m.keys()) == {"task_id", "seed", "train_subjects", "val_subjects", "test_subjects"}
    assert m["task_id"] == "toy.task"
    assert m["seed"] == 0
    assert set(m["train_subjects"]) | set(m["val_subjects"]) | set(m["test_subjects"]) == set(
        _toy_subjects(50)
    )
