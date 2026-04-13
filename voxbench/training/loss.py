"""
Loss function factory and class-weight computation.
"""
import numpy as np
import torch
import torch.nn.functional as F


def compute_pos_weights(
    labels: np.ndarray,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    """Inverse class-frequency weights, clipped to [clip_min, clip_max].

    Args:
        labels:   Binary label matrix [N, D] (recording-level, training split).
        clip_min: Lower bound (typically 1.0 — no down-weighting of majority).
        clip_max: Upper bound (typically 30.0 — cap rare-disease up-weighting).

    Returns:
        Float tensor [D] to pass as pos_weight to BCE loss.
    """
    n_pos = labels.sum(axis=0).clip(min=1)
    n_neg = (labels.shape[0] - labels.sum(axis=0)).clip(min=1)
    return torch.from_numpy(
        np.clip(n_neg / n_pos, clip_min, clip_max).astype(np.float32)
    )


def make_loss_fn(
    pos_weights:     torch.Tensor,
    label_smoothing: float,
    focal_gamma:     float,
):
    """Return a (logits, labels) → scalar loss function.

    Combines:
      • Weighted BCE for class imbalance (pos_weights).
      • Label smoothing: targets shifted 0 → eps/2, 1 → 1 − eps/2.
      • Optional focal modulation (focal_gamma = 0 → standard weighted BCE).

    The returned function moves pos_weights to the same device as logits
    automatically, so it works with both CPU and GPU batches.
    """
    _eps   = label_smoothing
    _gamma = focal_gamma
    _pw    = pos_weights  # kept on CPU; moved to device inside

    def loss_fn(
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        smooth = labels * (1 - _eps) + 0.5 * _eps       # label smoothing
        pw = _pw.to(logits.device)
        if _gamma == 0.0:
            if label_mask is None:
                return F.binary_cross_entropy_with_logits(logits, smooth, pos_weight=pw)
            raw = F.binary_cross_entropy_with_logits(
                logits, smooth, pos_weight=pw, reduction="none"
            )
            mask = label_mask.to(logits.device).float()
            denom = mask.sum().clamp_min(1.0)
            return (raw * mask).sum() / denom
        # Focal-weighted BCE: scale by (1 − p_t)^γ
        bce = F.binary_cross_entropy_with_logits(
            logits, smooth, pos_weight=pw, reduction="none"
        )
        p  = torch.sigmoid(logits)
        pt = smooth * p + (1 - smooth) * (1 - p)       # probability of correct outcome
        raw = ((1 - pt) ** _gamma) * bce
        if label_mask is None:
            return raw.mean()
        mask = label_mask.to(logits.device).float()
        denom = mask.sum().clamp_min(1.0)
        return (raw * mask).sum() / denom

    return loss_fn
