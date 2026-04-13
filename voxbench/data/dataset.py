"""VoxClinBench-Base preprocessing — VoiceDataset, normalization
statistics, and DataLoader factory.

**Scope**: preprocessing for our reference VoxClinBench-Base model.
Submitters using their own pipeline should ignore this file and
output the two-column ``subject_id,predicted_prob`` CSV directly.
"""
import json
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from voxbench.config import CONFIG, MODALITY_KEYS, N_STATIC_FEATURES


class VoiceDataset(Dataset):
    """Lazy HDF5 reader; one file handle opened per DataLoader worker."""

    def __init__(
        self,
        h5_path:     str,
        uid_list:    list[str],
        labels:      np.ndarray,       # [N, len(DISEASE_LIST)]
        norm_stats:  dict,
        enabled:     dict[str, bool],  # tier modality mask
        disease_idx: list[int],        # indices into the full label vector
        training:    bool = False,
        pids:        list | None = None,
        task_names:  list | None = None,
        n_static_features: int = N_STATIC_FEATURES,
    ):
        self.h5_path     = h5_path
        self.uids        = uid_list
        self.labels      = labels
        self.norm_stats  = norm_stats
        self.enabled     = enabled
        self.disease_idx = disease_idx
        self.training    = training
        self.pids        = pids       if pids       is not None else []
        self.task_names  = task_names if task_names is not None else []
        self.n_static_features = int(n_static_features)
        self._h5: h5py.File | None = None

    @staticmethod
    def _spec_augment(
        arr:             torch.Tensor,  # [..., F, T]
        n_time_masks:    int,
        time_mask_ratio: float,
        n_freq_masks:    int,
        freq_mask_ratio: float,
    ) -> torch.Tensor:
        """SpecAugment: random time + frequency masking on a 2-D feature map."""
        arr = arr.clone()
        F, T = arr.shape[-2], arr.shape[-1]
        t_max_w = max(1, int(T * time_mask_ratio))
        f_max_w = max(1, int(F * freq_mask_ratio))
        for _ in range(n_time_masks):
            w = random.randint(0, t_max_w)
            s = random.randint(0, max(0, T - w))
            arr[..., :, s:s + w] = 0.0
        for _ in range(n_freq_masks):
            w = random.randint(0, f_max_w)
            s = random.randint(0, max(0, F - w))
            arr[..., s:s + w, :] = 0.0
        return arr

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx: int) -> dict:
        self._open()
        uid   = self.uids[idx]
        grp   = self._h5[f"recordings/{uid}"]
        avail = grp["available_mask"][:].copy()  # [len(MODALITY_KEYS)] bool

        # Override: mask out modalities disabled for this tier
        for i, key in enumerate(MODALITY_KEYS):
            if not self.enabled.get(key, False):
                avail[i] = False

        t          = CONFIG["T_MAX"]
        std_floor  = float(CONFIG.get("norm_std_floor", 1e-3))
        clip_val   = float(CONFIG.get("feature_clip",   0.0))

        def _load(key: str, zeros_shape: tuple) -> tuple[torch.Tensor, int]:
            """Load one modality, normalise, and return (tensor, actual_length).

            actual_length is the last non-zero frame index + 1 detected *before*
            normalisation so that zero-padding is not included in stats pooling.
            For unavailable modalities it is 0.
            """
            mod_i = MODALITY_KEYS.index(key)
            if avail[mod_i] and key in grp:
                arr = grp[key][:].astype(np.float32)
                # Mel is stored as linear power; log1p-transform to match log-magnitude
                # scale of the spectrogram branch and reduce the extreme std range.
                if key == "mel":
                    arr = np.log1p(arr)
                # Detect actual (pre-pad) length from trailing zeros
                if arr.ndim >= 2:
                    nz = np.where(arr.any(axis=tuple(range(arr.ndim - 1))))[0]
                    act_len = int(nz[-1] + 1) if len(nz) > 0 else arr.shape[-1]
                else:
                    act_len = arr.shape[-1]  # static: no time axis
                # Normalise
                if key in self.norm_stats:
                    mn  = self.norm_stats[key]["mean"]
                    std = self.norm_stats[key]["std"]
                    if arr.ndim == 2:
                        mn  = mn[:, None]
                        std = std[:, None]
                    arr = (arr - mn) / np.maximum(std, std_floor)
                    if clip_val > 0:
                        arr = np.clip(arr, -clip_val, clip_val)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                return torch.from_numpy(arr), act_len
            return torch.zeros(zeros_shape, dtype=torch.float32), 0

        label_tier = self.labels[idx][self.disease_idx]

        spec,   _             = _load("spec",   (201, t))
        mfcc,   _             = _load("mfcc",   (60,  t))
        mel,    _             = _load("mel",    (60,  t))
        ppg,    act_len_ppg   = _load("ppg",    (40,  t))
        ema,    act_len_ema   = _load("ema",    (12,  t))
        pros,   act_len_pros  = _load("pros",   (3,   t))
        static, _             = _load("static", (self.n_static_features,))

        # Add channel dim for 2-D modalities fed into EfficientNet / ResNet.
        # PPG stays as [40, T] — BranchD uses Conv1D with 40 input channels.
        spec = spec.unsqueeze(0)   # [1, 201, T]
        mfcc = mfcc.unsqueeze(0)   # [1,  60, T]
        mel  = mel.unsqueeze(0)    # [1,  60, T]
        # ppg: [40, T]             — no unsqueeze; Conv1D branch expects [B, 40, T]

        if self.training:
            aug  = CONFIG.get("augmentation", {})
            sa   = VoiceDataset._spec_augment
            args = (
                aug.get("n_time_masks",    2),
                aug.get("time_mask_ratio", 0.20),
                aug.get("n_freq_masks",    2),
                aug.get("freq_mask_ratio", 0.15),
            )
            spec = sa(spec, *args)
            mel  = sa(mel,  *args)
            mfcc = sa(mfcc, *args)
            ns = aug.get("noise_std", 0.05)
            if ns > 0:
                mel = mel + torch.randn_like(mel) * ns
                mfcc = mfcc + torch.randn_like(mfcc) * ns

        return {
            "spec":         spec,
            "mfcc":         mfcc,
            "mel":          mel,
            "ppg":          ppg,
            "ema":          ema,
            "pros":         pros,
            "static":       static,
            "available":    torch.from_numpy(avail),
            "label":        torch.from_numpy(label_tier.astype(np.float32)),
            "pid":          torch.tensor(self.pids[idx] if self.pids else -1,
                                         dtype=torch.long),
            "task_name":    self.task_names[idx] if self.task_names else "",
            "act_len_ppg":  torch.tensor(act_len_ppg,  dtype=torch.long),
            "act_len_ema":  torch.tensor(act_len_ema,  dtype=torch.long),
            "act_len_pros": torch.tensor(act_len_pros, dtype=torch.long),
        }


def compute_normalization_stats(
    h5_path:    str,
    train_uids: list[str],
    enabled:    dict[str, bool],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-modality mean / std from training recordings only.

    NaN / Inf values are excluded per feature dimension when accumulating
    moments so corrupted frames do not bias the statistics.
    """
    stats: dict[str, dict[str, np.ndarray]] = {}
    silent = not os.isatty(1)

    with h5py.File(h5_path, "r") as hf:
        recs = hf["recordings"]
        for mod in MODALITY_KEYS:
            if not enabled.get(mod, False):
                continue
            accum_sum = accum_sq = count = None

            for uid in tqdm(train_uids, desc=f"norm/{mod}", leave=False, disable=silent):
                if uid not in recs or mod not in recs[uid]:
                    continue
                arr  = recs[uid][mod][:].astype(np.float64)
                if mod == "mel":
                    arr = np.log1p(arr)  # match _load() transform
                flat = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr.reshape(-1, 1)
                valid = np.isfinite(flat)
                safe  = np.where(valid, flat, 0.0)
                s  = safe.sum(axis=1)
                sq = (safe ** 2).sum(axis=1)
                n  = valid.sum(axis=1).astype(np.float64)
                accum_sum = s  if accum_sum is None else accum_sum + s
                accum_sq  = sq if accum_sq  is None else accum_sq  + sq
                count     = n  if count     is None else count     + n

            if accum_sum is None:
                continue

            valid_dim = count > 0
            mean = np.zeros_like(accum_sum)
            mean[valid_dim] = accum_sum[valid_dim] / count[valid_dim]
            var = np.zeros_like(accum_sum)
            var[valid_dim] = accum_sq[valid_dim] / count[valid_dim] - mean[valid_dim] ** 2
            std = np.sqrt(np.maximum(var, 1e-12))
            mean[~valid_dim] = 0.0
            std[~valid_dim]  = 1.0
            stats[mod] = {
                "mean": np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                "std":  np.nan_to_num(std,  nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32),
            }
    return stats


def make_dataloaders(
    h5_path:     str,
    norm_stats:  dict,
    enabled:     dict[str, bool],
    disease_idx: list[int],
    batch_size:  int,
    n_workers:   int,
    pos_weights: torch.Tensor | None = None,
    n_static_features: int = N_STATIC_FEATURES,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train / val / test DataLoaders from the HDF5 file.

    Returns (train_loader, val_loader, test_loader, train_uids).
    """
    with h5py.File(h5_path, "r") as hf:
        meta       = hf["metadata"]
        uids       = [u.decode() for u in meta["uids"][:]]
        pids       = meta["pids"][:].tolist()
        labels     = meta["labels"][:]
        task_names = (
            [t.decode() for t in meta["task_names"][:]]
            if "task_names" in meta else [""] * len(uids)
        )
        splits = json.loads(meta.attrs["splits_json"])

    def _filter(pid_set):
        idx = [i for i, p in enumerate(pids) if p in pid_set]
        return ([uids[i] for i in idx], labels[idx],
                [pids[i] for i in idx], [task_names[i] for i in idx])

    train_uids, tl, train_pids, train_tasks = _filter(set(splits["train"]))
    val_uids,   vl, val_pids,   val_tasks   = _filter(set(splits["val"]))
    test_uids,  sl, test_pids,  test_tasks  = _filter(set(splits["test"]))

    print(f"Split — train: {len(train_uids)}, val: {len(val_uids)}, test: {len(test_uids)}")

    def _build_sampler(lbl: np.ndarray) -> WeightedRandomSampler:
        """Up-sample rare-disease recordings using max pos_weight of positive labels."""
        pw      = pos_weights.numpy()
        weights = np.ones(len(lbl), dtype=np.float32)
        for i, row in enumerate(lbl[:, disease_idx]):
            pos = row.astype(bool)
            if pos.any():
                weights[i] = pw[pos].max()
        return WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=len(weights), replacement=True
        )

    def _loader(uid_l, lbl, task_l, shuffle, training=False, pid_l=None):
        ds = VoiceDataset(h5_path, uid_l, lbl, norm_stats, enabled, disease_idx,
                          training=training, pids=pid_l, task_names=task_l,
                          n_static_features=n_static_features)
        sampler = _build_sampler(lbl) if (training and pos_weights is not None) else None
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=True,
            persistent_workers=(n_workers > 0),
        )

    return (
        _loader(train_uids, tl, train_tasks, True,  training=True, pid_l=train_pids),
        _loader(val_uids,   vl, val_tasks,   False, pid_l=val_pids),
        _loader(test_uids,  sl, test_tasks,  False, pid_l=test_pids),
        train_uids,
    )
