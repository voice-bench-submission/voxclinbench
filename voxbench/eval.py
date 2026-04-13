"""Subject-level evaluation harness.

Computes AUROC, AUPRC, bootstrap 95% CI, and DeLong significance tests
from per-subject probability vectors. Holm-Bonferroni correction is
applied within each task family.

All functions are pure: they take arrays in and return new arrays / new
dicts out. No mutation of inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260412
ALPHA = 0.05


@dataclass(frozen=True)
class EvalResult:
    task_id: str
    n_test: int
    auroc: float
    auprc: float | None
    ci95: tuple[float, float]
    subject_probs: tuple[dict, ...]


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Thin wrapper so we can swap sklearn out in restricted envs.
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_prob))


def _pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_prob))


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                 resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Subject-level bootstrap CI for AUROC. Returns (lo, hi) at 95%."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            scores[i] = np.nan
            continue
        scores[i] = _roc_auc(y_true[idx], y_prob[idx])
    scores = scores[~np.isnan(scores)]
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_task(task_id: str, subject_probs: Sequence[dict],
                  *, include_auprc: bool = True) -> EvalResult:
    """Score one task from its per-subject probability list."""
    y_true = np.asarray([s["y_true"] for s in subject_probs], dtype=int)
    y_prob = np.asarray([s["y_prob"] for s in subject_probs], dtype=float)
    auroc = _roc_auc(y_true, y_prob)
    auprc = _pr_auc(y_true, y_prob) if include_auprc else None
    ci = bootstrap_ci(y_true, y_prob)
    return EvalResult(
        task_id=task_id,
        n_test=len(y_true),
        auroc=auroc,
        auprc=auprc,
        ci95=ci,
        subject_probs=tuple(subject_probs),
    )


def holm_bonferroni(pvals: Sequence[float],
                    alpha: float = ALPHA) -> tuple[bool, ...]:
    """Return per-hypothesis rejection flags under Holm-Bonferroni."""
    m = len(pvals)
    order = np.argsort(pvals)
    rejected_sorted = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvals[idx] <= threshold:
            rejected_sorted[rank] = True
        else:
            break
    rejected = [False] * m
    for rank, idx in enumerate(order):
        rejected[idx] = rejected_sorted[rank]
    return tuple(rejected)


def paired_bootstrap_p(y_true: np.ndarray,
                       y_pred1: np.ndarray,
                       y_pred2: np.ndarray,
                       resamples: int = BOOTSTRAP_RESAMPLES,
                       seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Paired-subject bootstrap two-sided p-value for AUC1 - AUC2.

    Returns (auc_diff, p_value). Useful as a non-parametric sanity check
    against the analytic DeLong kernel.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred1 = np.asarray(y_pred1, dtype=np.float64).ravel()
    y_pred2 = np.asarray(y_pred2, dtype=np.float64).ravel()
    observed = _roc_auc(y_true, y_pred1) - _roc_auc(y_true, y_pred2)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = _roc_auc(y_true[idx], y_pred1[idx]) - _roc_auc(
            y_true[idx], y_pred2[idx]
        )
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float(observed), 1.0
    centered = diffs - diffs.mean()
    p = float(2.0 * min((centered <= -abs(observed)).mean(),
                        (centered >= abs(observed)).mean()))
    p = min(1.0, max(p, 1.0 / len(diffs)))
    return float(observed), p


def _midrank(x: np.ndarray) -> np.ndarray:
    """Compute mid-ranks of the values in x (ties averaged).

    O(N log N) via argsort. Matches the Tx/Ty mid-rank convention of
    Sun & Xu 2014, eq. (3).
    """
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        # mid-rank (1-indexed average of tied ranks)
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray,
                 label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong kernel from Sun & Xu 2014.

    Parameters
    ----------
    predictions_sorted_transposed : ndarray, shape [k, N]
        k classifier prediction vectors, with positives first (m cols)
        then negatives (n cols). N = m + n.
    label_1_count : int
        m, the number of positive samples.

    Returns
    -------
    aucs : ndarray [k]
    delongcov : ndarray [k, k]
        Covariance matrix of AUC estimators (Eq. 7 of Sun & Xu).
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r] = _midrank(positive_examples[r])
        ty[r] = _midrank(negative_examples[r])
        tz[r] = _midrank(predictions_sorted_transposed[r])
    # AUC via Eq. (2): (sum Tz_X - m(m+1)/2) / (m*n)
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    # V components (Eq. 4-5)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    # np.cov returns 0-d for k=1; normalise to 2-d
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def _calc_pvalue(aucs: np.ndarray, sigma: np.ndarray) -> float:
    """Two-sided p-value for H0: AUC1 = AUC2 under asymptotic normality."""
    l = np.array([[1.0, -1.0]])
    diff = float((l @ aucs).item())
    var = float((l @ sigma @ l.T).item())
    if var <= 0.0:
        # Degenerate: identical predictions (or near-identical)
        return 1.0 if diff == 0.0 else 0.0
    z = diff / math.sqrt(var)
    # two-sided p via erfc for numerical stability in tails
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(p)


def _compute_ground_truth_statistics(
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, int]:
    order = (-ground_truth).argsort(kind="mergesort")
    label_1_count = int((ground_truth == 1).sum())
    return order, label_1_count


def delong_p(y_true: np.ndarray,
             y_pred1: np.ndarray,
             y_pred2: np.ndarray) -> tuple[float, float]:
    """Paired DeLong test (Sun & Xu 2014) comparing two AUROCs.

    Parameters
    ----------
    y_true : array of 0/1 labels, shape [N]
    y_pred1, y_pred2 : paired score vectors, shape [N]

    Returns
    -------
    (auc_diff, p_value) : tuple[float, float]
        auc_diff = AUC1 - AUC2
        p_value  = two-sided p under H0: AUC1 = AUC2

    Raises
    ------
    ValueError : on NaN inputs, shape mismatch, or single-class labels.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred1 = np.asarray(y_pred1, dtype=np.float64).ravel()
    y_pred2 = np.asarray(y_pred2, dtype=np.float64).ravel()
    if y_true.shape != y_pred1.shape or y_true.shape != y_pred2.shape:
        raise ValueError("y_true, y_pred1, y_pred2 must have identical shape")
    if np.isnan(y_pred1).any() or np.isnan(y_pred2).any() or np.isnan(y_true).any():
        raise ValueError("DeLong inputs contain NaN")
    uniq = np.unique(y_true)
    if not np.array_equal(np.sort(uniq), np.array([0, 1])):
        raise ValueError(
            f"DeLong requires both classes present; got labels {uniq.tolist()}"
        )
    order, m = _compute_ground_truth_statistics(y_true.astype(np.int64))
    preds_sorted = np.vstack((y_pred1, y_pred2))[:, order]
    aucs, delongcov = _fast_delong(preds_sorted, m)
    p = _calc_pvalue(aucs, delongcov)
    return float(aucs[0] - aucs[1]), p


def load_submission(path: str | Path) -> dict:
    """Load a submission file.

    Supports two formats:

    1. **JSON** (full submission with metadata): must contain
       ``task_id``, ``seed``, ``subject_probs`` (dict of subject_id
       -> probability). Canonical submission format for the hosted
       leaderboard.

    2. **CSV** (lightweight per-seed predictions, as shipped in
       ``predictions/``): exactly two columns
       ``subject_id,predicted_prob``. Filename is parsed as
       ``<task_id>.seed<s>.<baseline>.csv`` to recover metadata.

    Both paths return a dict with the three canonical keys.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import csv as _csv
        import re as _re

        m = _re.match(r"^(.+)\.seed(\d+)\.(.+)\.csv$", path.name)
        task_id, seed = (m.group(1), int(m.group(2))) if m else (path.stem, 0)
        subject_probs: dict[str, float] = {}
        with path.open(newline="") as f:
            rdr = _csv.DictReader(f)
            if rdr.fieldnames is None or "subject_id" not in rdr.fieldnames:
                raise ValueError(
                    f"CSV submission {path} must have a 'subject_id' column; "
                    f"found {rdr.fieldnames}"
                )
            prob_col = next(
                (c for c in rdr.fieldnames
                 if c in ("predicted_prob", "prob", "probability", "pred")),
                None,
            )
            if prob_col is None:
                raise ValueError(
                    f"CSV submission {path} must have one of "
                    f"(predicted_prob/prob/probability/pred); "
                    f"found {rdr.fieldnames}"
                )
            for row in rdr:
                sid = row["subject_id"]
                try:
                    subject_probs[sid] = float(row[prob_col])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"CSV submission {path}: non-numeric probability "
                        f"for subject {sid!r}: {row[prob_col]!r}"
                    )
        return {"task_id": task_id, "seed": seed, "subject_probs": subject_probs}

    with path.open() as f:
        payload = json.load(f)
    required = {"task_id", "seed", "subject_probs"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Submission {path} missing fields: {sorted(missing)}")
    return payload
