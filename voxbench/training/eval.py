"""
Evaluation utilities: patient-level AUROC and early stopping.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_auroc(
    model:                nn.Module,
    loader:               DataLoader,
    device:               torch.device,
    disease_names:        list[str],
    task_filter_keywords: dict[str, list[str]] | None = None,
    patient_filter_map:   dict[str, set[int]] | None = None,
    min_support_patients: int = 0,
) -> tuple[float, list[float]]:
    """Compute AUROC with patient-level probability aggregation.

    All recordings of a patient are averaged into a single probability score
    before roc_auc_score. This removes within-patient correlation (~38 recordings
    share the same label) and gives a more stable, clinically meaningful metric.

    Args:
        task_filter_keywords: {disease: [keyword, ...]} — for a given disease,
            only recordings whose task_name contains at least one keyword are used.
            Falls back to all recordings if none match. Pass None or {} to use
            all recordings for all diseases.
        patient_filter_map: {disease: {pid, ...}} — optional task-specific
            patient eligibility set. When provided, AUROC for a disease is
            computed only on the corresponding patient subset.
        min_support_patients: minimum number of positive patients required for a
            disease to contribute to the reported macro AUROC. Diseases below this
            threshold are still evaluated individually but excluded from the macro
            average. Use this for early-stopping to avoid noisy macro from rare
            diseases with few val patients (e.g. ALS=3, Laryngeal Cancer=4).

    Returns:
        (macro_auroc, per_disease_aurocs)  — NaN for diseases with no positives or
        below min_support_patients.
    """
    model.eval()
    all_logits, all_labels, all_pids, all_tasks = [], [], [], []

    for batch in loader:
        logits = model(
            batch["spec"].to(device),
            batch["mfcc"].to(device),    batch["mel"].to(device), batch["ppg"].to(device),
            batch["ema"].to(device),     batch["pros"].to(device),
            batch["static"].to(device),  batch["available"].to(device),
            act_len_ppg=batch["act_len_ppg"].to(device),
            act_len_ema=batch["act_len_ema"].to(device),
            act_len_pros=batch["act_len_pros"].to(device),
        )
        all_logits.append(logits.cpu().float())
        all_labels.append(batch["label"])
        all_pids.extend(batch["pid"].tolist())
        all_tasks.extend([str(t).lower() for t in batch["task_name"]])

    logits  = torch.cat(all_logits).numpy()
    labels  = torch.cat(all_labels).numpy()

    if not np.isfinite(logits).all():
        bad = np.size(logits) - int(np.isfinite(logits).sum())
        raise RuntimeError(f"[evaluate_auroc] Non-finite logits: {bad} elements.")

    probs   = 1.0 / (1.0 + np.exp(-logits))
    pid_arr = np.array(all_pids)
    filter_cfg = task_filter_keywords or {}
    patient_filters = patient_filter_map or {}

    def _task_mask(disease: str) -> np.ndarray:
        keys = [k.lower() for k in filter_cfg.get(disease, [])]
        if not keys:
            return np.ones(len(all_tasks), dtype=bool)
        return np.array([any(k in t for k in keys) for t in all_tasks], dtype=bool)

    def _patient_mask(disease: str) -> np.ndarray:
        keep = patient_filters.get(disease)
        if not keep:
            return np.ones(len(unique_pids), dtype=bool)
        return np.array([pid in keep for pid in unique_pids], dtype=bool)

    # Patient-level aggregation
    unique_pids = np.unique(pid_arr)
    n_pat = len(unique_pids)
    n_dis = labels.shape[1]
    pat_probs  = np.zeros((n_pat, n_dis), dtype=np.float32)
    pat_labels = np.zeros((n_pat, n_dis), dtype=np.float32)

    for d in range(n_dis):
        d_task_mask = _task_mask(disease_names[d])
        for j, pid in enumerate(unique_pids):
            pid_mask = pid_arr == pid
            use_mask = pid_mask & d_task_mask
            if not use_mask.any():
                use_mask = pid_mask   # fallback: use all recordings
            pat_probs[j, d]  = probs[use_mask, d].mean()
            pat_labels[j, d] = labels[pid_mask][0, d]

    per_disease = []
    for d in range(n_dis):
        use = _patient_mask(disease_names[d])
        y, p = pat_labels[use, d], pat_probs[use, d]
        n_pos = int(y.sum())
        n_neg = int((1 - y).sum())
        if n_pos == 0 or n_neg == 0:
            per_disease.append(float("nan"))
        else:
            per_disease.append(float(roc_auc_score(y, p)))

    # Macro uses only diseases with sufficient patient support (avoids noisy
    # AUROCs from diseases with very few val patients driving early stopping)
    valid = [
        a for d, a in enumerate(per_disease)
        if not math.isnan(a)
        and int(pat_labels[_patient_mask(disease_names[d]), d].sum()) >= max(min_support_patients, 1)
    ]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_disease


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best     = -float("inf")
        self.counter  = 0

    def step(self, metric: float) -> bool:
        """Return True if training should stop."""
        if metric > self.best:
            self.best    = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
