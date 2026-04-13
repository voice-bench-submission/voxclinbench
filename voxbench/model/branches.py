"""
Lightweight CNN branches for time-series and static modalities,
plus masked statistics-pooling helpers.

Statistics pooling (mean + std) replaces global average pooling:
  - std captures tremor (EMA), dysarthria (prosodic), aperiodicity (PPG)
  - pooling is masked by actual recording length so zero-padded frames
    do not bias the statistics
"""
import torch
import torch.nn as nn


# ── Masked statistics pooling ─────────────────────────────────────────────────

def _stats_pool_1d(h: torch.Tensor, act_len: torch.Tensor | None = None) -> torch.Tensor:
    """Mean + std pooling over time for [B, C, T].

    Args:
        h:       Feature map [B, C, T].
        act_len: Valid frame counts [B]; if None, uses the full sequence.

    Returns:
        [B, 2C] — concatenated mean and std.
    """
    B, C, T = h.shape
    if act_len is not None:
        mask = torch.arange(T, device=h.device).unsqueeze(0) < act_len.unsqueeze(1)  # [B, T]
        mf   = mask.float().unsqueeze(1)                    # [B, 1, T]
        n    = act_len.clamp(min=1).float()                  # [B]
        mean = (h * mf).sum(-1) / n.unsqueeze(1)             # [B, C]
        var  = ((h - mean.unsqueeze(-1)) ** 2 * mf).sum(-1) / n.unsqueeze(1)
        std  = torch.sqrt(var.clamp(min=1e-6))
    else:
        mean = h.mean(-1)
        std  = torch.sqrt(h.var(-1, unbiased=False).clamp(min=1e-6))
    return torch.cat([mean, std], dim=1)  # [B, 2C]


def _stats_pool_2d(h: torch.Tensor, act_len: torch.Tensor | None = None) -> torch.Tensor:
    """Mean + std pooling over (freq × time) for [B, C, F, T].

    Time axis is masked by act_len; frequency axis is always fully pooled.

    Returns:
        [B, 2C]
    """
    B, C, F, T = h.shape
    if act_len is not None:
        mask = torch.arange(T, device=h.device).unsqueeze(0) < act_len.unsqueeze(1)  # [B, T]
        mf   = mask.float().unsqueeze(1).unsqueeze(2)          # [B, 1, 1, T]
        n    = (F * act_len.clamp(min=1)).float()               # [B]
        mean = (h * mf).sum((-2, -1)) / n.unsqueeze(1)         # [B, C]
        var  = ((h - mean.unsqueeze(-1).unsqueeze(-1)) ** 2 * mf).sum((-2, -1)) / n.unsqueeze(1)
        std  = torch.sqrt(var.clamp(min=1e-6))
    else:
        mean = h.mean((-2, -1))
        std  = torch.sqrt(h.var((-2, -1), unbiased=False).clamp(min=1e-6))
    return torch.cat([mean, std], dim=1)  # [B, 2C]


# ── Modality branches ─────────────────────────────────────────────────────────

class BranchD(nn.Module):
    """PPG branch: [B, 40, T] → [B, 512]

    PPG's 40 phoneme dimensions are categorically independent — they represent
    distinct phoneme classes, not spatially adjacent frequency bins.  Conv2D
    over the phoneme axis has no spatial meaning; Conv1D treats the 40 phonemes
    as input channels and slides only over time, which is correct.

    3 × Conv1D layers + masked 1-D stats pooling (256 channels → 512 with mean+std).
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(40,  64,  3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, act_len: torch.Tensor | None = None) -> torch.Tensor:
        return self.drop(_stats_pool_1d(self.convs(x), act_len))


class BranchE(nn.Module):
    """EMA branch: [B, 12, T] → [B, 256]

    3 × Conv1D layers + masked 1-D stats pooling (128 channels → 256 with mean+std).
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(12, 32,  3, padding=1), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32, 64,  3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, act_len: torch.Tensor | None = None) -> torch.Tensor:
        return self.drop(_stats_pool_1d(self.convs(x), act_len))


class BranchF(nn.Module):
    """Prosodic branch: [B, 3, T] → [B, 128]

    3 × Conv1D layers + masked 1-D stats pooling (64 channels → 128 with mean+std).
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(3,  16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, act_len: torch.Tensor | None = None) -> torch.Tensor:
        return self.drop(_stats_pool_1d(self.convs(x), act_len))


class BranchG(nn.Module):
    """Static feature branch: [B, 131] → [B, 64]

    2-layer MLP with LayerNorm.
    """
    def __init__(self, in_features: int = 131, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),          nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
