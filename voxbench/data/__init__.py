"""Data utilities for VoxClinBench.

Public API:
    make_splits(task_id, seed, subjects, labels=None) -> dict
        Deterministic participant-wise 80/20 (stratified if labels given)
        split. Reviewers call this after `voxbench fetch <corpus>` to
        regenerate train/val subject lists that match the test set shipped
        in ``splits/<task_id>.seed<seed>.json``.
"""
from voxbench.data.splits import make_splits

__all__ = ["make_splits"]
