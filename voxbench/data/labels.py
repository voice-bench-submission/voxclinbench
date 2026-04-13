"""
Label loading, participant splits, and static feature loading.
"""
import json
import os

import numpy as np
import pandas as pd

from voxbench.config import DISEASE_LIST, DISEASE_FILES


def _pid_to_int(pid) -> int:
    """Normalise participant_id to int.

    Handles zero-padded strings ('001573') and raw integers (502326) alike.
    """
    return int(str(pid).lstrip("0") or "0")


def load_labels(data_root: str) -> dict[int, np.ndarray]:
    """Return {participant_id (int): label vector [len(DISEASE_LIST)] float32}."""
    diag_dir = os.path.join(data_root, "phenotype", "diagnosis")

    pos_sets: dict[str, set[int]] = {}
    for name, fname in DISEASE_FILES.items():
        path = os.path.join(diag_dir, fname)
        df = pd.read_csv(path, sep="\t", usecols=["participant_id"])
        pos_sets[name] = {_pid_to_int(p) for p in df["participant_id"]}

    part_path = os.path.join(data_root, "phenotype", "enrollment", "participant.tsv")
    all_pids = pd.read_csv(part_path, sep="\t")["participant_id"].tolist()

    return {
        _pid_to_int(pid): np.array(
            [1.0 if _pid_to_int(pid) in pos_sets[d] else 0.0 for d in DISEASE_LIST],
            dtype=np.float32,
        )
        for pid in all_pids
    }


def build_task_manifest(
    data_root: str,
    task_names: list[str],
    seed: int = 42,
    test_frac: float = 0.20,
    val_frac_within_train: float = 0.125,
    global_splits: dict[str, list[int]] | None = None,
) -> dict:
    """Build task-specific binary patient cohorts for fair baseline comparison.

    If *global_splits* is given (a dict with keys "train", "val", "test", each
    a list of patient IDs), every patient is assigned to the same partition as
    in the global split.  This guarantees that a two-stage pipeline (e.g.
    Seasoned Learning) never leaks Stage-1 training data into Stage-2
    validation or test sets.

    Without *global_splits* the function falls back to its original behaviour:
    an independent random split per disease cohort.
    """
    labels_by_pid = load_labels(data_root)
    pids = np.array(sorted(labels_by_pid))
    label_mat = np.stack([labels_by_pid[int(pid)] for pid in pids]).astype(np.float32)
    disease_to_idx = {name: i for i, name in enumerate(DISEASE_LIST)}
    any_other_dx = label_mat.sum(axis=1) > 0
    rng = np.random.default_rng(seed=seed)

    if global_splits is not None:
        train_set = set(int(p) for p in global_splits["train"])
        val_set = set(int(p) for p in global_splits["val"])
        test_set = set(int(p) for p in global_splits["test"])

    def _split_group_global(group: np.ndarray) -> dict[str, list[int]]:
        """Assign each patient to the same partition as the global split."""
        group = np.array(sorted(int(x) for x in group), dtype=np.int64)
        return {
            "train": [int(p) for p in group if p in train_set],
            "val":   [int(p) for p in group if p in val_set],
            "test":  [int(p) for p in group if p in test_set],
        }

    def _split_group_random(group: np.ndarray) -> dict[str, list[int]]:
        """Original random split (standalone, no global constraint)."""
        group = np.array(sorted(int(x) for x in group), dtype=np.int64)
        if len(group) == 0:
            return {"train": [], "val": [], "test": []}
        shuffled = rng.permutation(group)
        n_test = max(1, int(round(len(shuffled) * test_frac)))
        n_test = min(n_test, max(len(shuffled) - 2, 1))
        test = shuffled[:n_test]
        remain = shuffled[n_test:]
        if len(remain) <= 1:
            return {"train": remain.tolist(), "val": [], "test": test.tolist()}
        n_val = int(round(len(remain) * val_frac_within_train))
        n_val = max(1, n_val) if len(remain) >= 4 else 0
        n_val = min(n_val, max(len(remain) - 1, 0))
        val = remain[:n_val]
        train = remain[n_val:]
        return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}

    _split_group = _split_group_global if global_splits is not None else _split_group_random

    tasks = {}
    for task_name in task_names:
        task_idx = disease_to_idx[task_name]
        pos = pids[label_mat[:, task_idx] == 1]
        neg_pool = pids[(label_mat[:, task_idx] == 0) & any_other_dx]
        if len(neg_pool) < len(pos):
            neg_pool = pids[label_mat[:, task_idx] == 0]

        if global_splits is not None:
            # Sample matched negatives independently within each partition
            # so that train/val/test boundaries are never crossed.
            neg_parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
            for split_name, split_set in [("train", train_set), ("val", val_set), ("test", test_set)]:
                split_pos = np.array([int(p) for p in pos if p in split_set])
                split_neg_pool = np.array([int(p) for p in neg_pool if p in split_set])
                n_want = len(split_pos)
                if n_want > 0 and len(split_neg_pool) > 0:
                    chosen = rng.choice(split_neg_pool, size=min(n_want, len(split_neg_pool)), replace=False)
                    neg_parts[split_name] = chosen.tolist()
                else:
                    neg_parts[split_name] = []
            neg_all = np.array(neg_parts["train"] + neg_parts["val"] + neg_parts["test"], dtype=np.int64)
        else:
            neg_all = (
                rng.choice(neg_pool, size=len(pos), replace=False)
                if len(pos) > 0
                else np.array([], dtype=np.int64)
            )

        pos_splits = _split_group(pos)
        if global_splits is not None:
            neg_splits = neg_parts  # already partitioned
        else:
            neg_splits = _split_group(neg_all)

        tasks[task_name] = {
            "task_idx": int(task_idx),
            "positive": pos_splits,
            "negative": neg_splits,
            "n_positive_patients": int(len(pos)),
            "n_negative_patients": int(len(neg_all)),
        }

    return {
        "version": "task_manifest_v1",
        "seed": int(seed),
        "test_frac": float(test_frac),
        "val_frac_within_train": float(val_frac_within_train),
        "global_splits_used": global_splits is not None,
        "tasks": tasks,
    }


def manifest_to_json(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True)


def manifest_from_json(payload: str) -> dict:
    return json.loads(payload)


def make_splits(
    data_root: str,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, list[int]]:
    """Deterministic participant-wise 70 / 15 / 15 split."""
    part_path = os.path.join(data_root, "phenotype", "enrollment", "participant.tsv")
    all_pids = sorted({_pid_to_int(p) for p in
                       pd.read_csv(part_path, sep="\t")["participant_id"]})

    rng = np.random.default_rng(seed=seed)
    shuffled = rng.permutation(all_pids)
    n_train = int(train_frac * len(shuffled))
    n_val   = int(val_frac   * len(shuffled))

    return {
        "train": shuffled[:n_train].tolist(),
        "val":   shuffled[n_train:n_train + n_val].tolist(),
        "test":  shuffled[n_train + n_val:].tolist(),
    }


def load_static_features(data_root: str) -> pd.DataFrame:
    """Load static_features.tsv; participant_id is normalised to int."""
    path = os.path.join(data_root, "features", "static_features.tsv")
    df = pd.read_csv(path, sep="\t")
    df["participant_id"] = df["participant_id"].apply(_pid_to_int)
    if "transcription" in df.columns:
        df = df.drop(columns=["transcription"])
    return df


def get_static_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature column names (exclude metadata columns)."""
    exclude = {"participant_id", "session_id", "task_name"}
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]
