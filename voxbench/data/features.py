"""VoxClinBench-Base preprocessing — array reconstruction helpers.

**Scope**: this module is the preprocessing we used to train the
reference VoxClinBench-Base 8-modality model. If your submission has
its own preprocessing, use yours — voxbench's only contract is the
two-column ``subject_id,predicted_prob`` CSV accepted by
``voxbench eval``.

Array reconstruction helpers for each modality. These functions
convert raw parquet rows into fixed-shape numpy arrays ready for
pad_or_truncate → HDF5 storage.
"""
import numpy as np

# Number of recording-level macro prosodic scalars appended to the 131-dim
# openSMILE/Praat static features → total static dim = 131 + N_MACRO_PROS = 137.
N_MACRO_PROS = 6


def pad_or_truncate(arr: np.ndarray, t_max: int) -> np.ndarray:
    """Truncate or zero-pad the last (time) axis to exactly t_max frames."""
    t = arr.shape[-1]
    if t >= t_max:
        return arr[..., :t_max]
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, t_max - t)]
    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


def reconstruct_2d(row, col: str) -> np.ndarray:
    """Stack object-array rows into [F, T] float32 (spec / mfcc / mel)."""
    return np.stack(row[col]).astype(np.float32)


def reconstruct_ppg(row) -> np.ndarray:
    """PPG: [40, T_ppg] → downsample ×2 → [40, T_ppg//2] (100 fps → 50 fps)."""
    mat = np.stack(row["ppgs"]).astype(np.float32)  # [40, T_ppg]
    return mat[:, ::2]


def reconstruct_ema(row) -> np.ndarray:
    """EMA: stored [T, 12] per row → transpose → [12, T]."""
    return np.stack(row["ema"]).astype(np.float32).T


def compute_macro_prosodic(
    loud_arr,
    period_arr,
    pitch_arr,
    period_thresh: float = 0.5,
) -> np.ndarray:
    """Compute N_MACRO_PROS (6) recording-level prosodic statistics.

    These capture long-range speech rhythm patterns that are clinically
    associated with psychiatric conditions (depression, ADHD, PTSD) and that
    the per-frame prosodic branch misses because it pools within T_MAX frames:

        0  phonation_ratio     fraction of frames with periodicity > threshold
                               depression → lower (more silence / breathiness)
        1  mean_voiced_dur     mean consecutive voiced-segment length (frames)
                               depression → shorter bursts
        2  std_voiced_dur      std of voiced-segment lengths (variability)
        3  mean_silence_dur    mean consecutive silence-run length (frames)
                               depression → longer pauses
        4  std_silence_dur     std of silence-run lengths
        5  n_pauses_per_100f   silence→voiced transitions per 100 frames
                               ADHD → more frequent switching; depression → fewer

    Returns float32 array of shape (N_MACRO_PROS,).  Returns zeros if input
    is empty or all-zero (e.g. unavailable modality).
    """
    period = np.asarray(period_arr, dtype=np.float32).ravel()
    T = len(period)
    if T == 0 or not np.any(period):
        return np.zeros(N_MACRO_PROS, dtype=np.float32)

    voiced = (period > period_thresh).astype(np.int8)
    phonation_ratio = float(voiced.mean())

    # Run-length encoding: find segment boundaries
    changes   = np.concatenate([[0], np.where(np.diff(voiced) != 0)[0] + 1, [T]])
    run_lens  = np.diff(changes).astype(np.float32)
    run_lbls  = voiced[changes[:-1]]          # 0=silence, 1=voiced, per run

    voiced_runs  = run_lens[run_lbls == 1]
    silence_runs = run_lens[run_lbls == 0]

    mean_voiced_dur  = float(voiced_runs.mean())  if len(voiced_runs)  > 0 else 0.0
    std_voiced_dur   = float(voiced_runs.std())   if len(voiced_runs)  > 1 else 0.0
    mean_silence_dur = float(silence_runs.mean()) if len(silence_runs) > 0 else 0.0
    std_silence_dur  = float(silence_runs.std())  if len(silence_runs) > 1 else 0.0

    # Transitions from silence to voice = number of pause-to-speech boundaries
    n_pauses        = int(np.sum((run_lbls[:-1] == 0) & (run_lbls[1:] == 1)))
    n_pauses_per_100f = 100.0 * n_pauses / T

    return np.array([
        phonation_ratio,
        mean_voiced_dur,
        std_voiced_dur,
        mean_silence_dur,
        std_silence_dur,
        n_pauses_per_100f,
    ], dtype=np.float32)


def reconstruct_prosodic(loud_arr, period_arr, pitch_arr) -> np.ndarray:
    """Stack 3 scalar time-series into [3, T], truncating to the shortest.

    Loudness / periodicity / pitch parquets can differ by ±1 frame for the
    same recording; truncating to min(T) avoids shape mismatches.
    """
    t = min(len(loud_arr), len(period_arr), len(pitch_arr))
    return np.stack([
        loud_arr[:t].astype(np.float32),
        period_arr[:t].astype(np.float32),
        pitch_arr[:t].astype(np.float32),
    ])  # [3, T]
