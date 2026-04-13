"""VoxClinBench data utilities.

Public API
----------
make_splits(task_id, seed, subjects, labels=None) -> Split
    Deterministic participant-wise 80/20 (stratified if labels given)
    split. Reviewers call this after ``voxbench fetch <corpus>`` to
    regenerate train/val subject lists that match the test set
    shipped in ``splits/<task_id>.seed<seed>.json``.

Preprocessing (our baseline's recipe)
-------------------------------------
``voxbench.data.features`` and ``voxbench.data.dataset`` implement
the 8-modality preprocessing pipeline used to train our VoxClinBench-
Base reference baseline (mel / MFCC / PPG / EMA / prosody / static /
SSL / clinical acoustic features, merged into an HDF5 store that
``voxbench.train`` consumes).

**This is our preprocessing. If your model has its own, use yours.**
The only contract voxbench enforces at submission time is the
two-column ``subject_id,predicted_prob`` CSV passed to
``voxbench eval``.
"""
from voxbench.data.splits import make_splits

__all__ = ["make_splits"]
