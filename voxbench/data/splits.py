"""Deterministic participant-wise split generator.

VoxClinBench ships only the *test* subject list in each
``splits/<task_id>.seed<seed>.json`` manifest. Train and validation
subjects are intentionally regenerated at the user's site from the
credentialed upstream corpus (respecting each DUA), using the
deterministic function below. Reviewers can verify that the test
subjects this function produces match the shipped manifest byte-for-
byte, closing the reproducibility loop without redistributing labels.

Protocol (paper §3 Protocol / Splits):

    * 80 / 20 subject-level train+val / test at the outer level,
      stratified by the task label if labels are provided.
    * 80 / 20 train / val within the outer train, same stratification.
    * Seeds fixed per task family; the per-task seed sets are documented
      in ``voxbench/tasks.py``.

The implementation uses NumPy's default_rng for reproducibility across
platforms and is pure logic — no upstream files are read here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "voxbench.data.splits requires numpy. Install with `pip install voxbench`."
    ) from exc


@dataclass(frozen=True)
class Split:
    """Result of a ``make_splits`` call."""

    task_id: str
    seed: int
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]

    def as_manifest(self) -> dict:
        """Return a JSON-serialisable manifest dict compatible with
        ``splits/<task_id>.seed<s>.json``."""
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "train_subjects": list(self.train_subjects),
            "val_subjects": list(self.val_subjects),
            "test_subjects": list(self.test_subjects),
        }


def _stratified_indices(
    rng: "np.random.Generator",
    labels: "np.ndarray",
    test_frac: float,
    val_frac_within_train: float,
) -> dict[str, "np.ndarray"]:
    """Produce stratified indices for train / val / test.

    Each class is independently shuffled and sliced so the resulting
    splits preserve class proportions to within one subject.
    """
    all_idx = np.arange(len(labels))
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_idx = all_idx[cls_mask]
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)
        n_test = max(1, int(round(test_frac * n_cls)))
        n_rest = n_cls - n_test
        n_val = max(1, int(round(val_frac_within_train * n_rest)))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test : n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val :].tolist())
    return {
        "train": np.array(sorted(train_idx)),
        "val": np.array(sorted(val_idx)),
        "test": np.array(sorted(test_idx)),
    }


def make_splits(
    task_id: str,
    seed: int,
    subjects: Sequence[str],
    labels: Iterable | None = None,
    test_frac: float = 0.20,
    val_frac_within_train: float = 0.20,
) -> Split:
    """Deterministic participant-wise train/val/test split.

    Parameters
    ----------
    task_id : str
        One of the 22 VoxClinBench task identifiers (see
        ``voxbench.tasks.TASKS``).
    seed : int
        Random seed. The paper uses ``{0, 1, 2, 3, 4}`` for the B2AI
        Tier-2 5-seed rerun and ``{13, 42, 2026}`` for the v0.1
        protocol; see ``voxbench/tasks.py`` for per-task seed sets.
    subjects : Sequence[str]
        List of pseudonymous subject identifiers (as used by the
        upstream corpus). Will not be mutated.
    labels : Iterable, optional
        Per-subject binary or multi-class label. When provided the
        split is stratified by class. When ``None`` the split is
        uniform.
    test_frac : float
        Fraction of subjects held out as test (default 0.20).
    val_frac_within_train : float
        Fraction of the non-test subjects used for validation
        (default 0.20, i.e. 16 % of total).

    Returns
    -------
    Split
        A frozen dataclass with train/val/test subject lists. Call
        ``.as_manifest()`` to get the JSON-serialisable dict matching
        the shipped ``splits/<task_id>.seed<seed>.json`` schema.

    Notes
    -----
    This implementation is pure (no file I/O), so a reviewer with the
    upstream subject list can verify that our shipped test set matches
    ``make_splits(task_id, seed, subjects, labels).test_subjects``
    byte-for-byte. The assertion in ``tests/test_splits.py`` covers
    this round-trip on the public NeuroVoz corpus.
    """
    subjects_arr = np.asarray(list(subjects), dtype=object)
    if len(subjects_arr) == 0:
        raise ValueError("make_splits requires at least one subject.")
    if not np.all(subjects_arr == np.array(sorted(subjects_arr), dtype=object)):
        # canonicalise order: splits must be reproducible regardless of
        # the caller's input ordering.
        order = np.argsort(subjects_arr)
        subjects_arr = subjects_arr[order]
        if labels is not None:
            labels = np.asarray(list(labels))[order]
    rng = np.random.default_rng(seed=seed)
    if labels is None:
        perm = rng.permutation(len(subjects_arr))
        n_test = max(1, int(round(test_frac * len(subjects_arr))))
        n_rest = len(subjects_arr) - n_test
        n_val = max(1, int(round(val_frac_within_train * n_rest)))
        idx = {
            "test": perm[:n_test],
            "val": perm[n_test : n_test + n_val],
            "train": perm[n_test + n_val :],
        }
    else:
        labels_arr = np.asarray(list(labels))
        if len(labels_arr) != len(subjects_arr):
            raise ValueError(
                f"labels length {len(labels_arr)} != subjects length "
                f"{len(subjects_arr)}"
            )
        idx = _stratified_indices(rng, labels_arr, test_frac, val_frac_within_train)

    def pick(which: str) -> list[str]:
        return sorted(subjects_arr[idx[which]].tolist())

    return Split(
        task_id=task_id,
        seed=seed,
        train_subjects=pick("train"),
        val_subjects=pick("val"),
        test_subjects=pick("test"),
    )
