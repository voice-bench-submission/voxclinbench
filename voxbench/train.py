"""VoxClinBench-Base training harness — 7-branch CNN-Transformer over 8 modalities.

**Scope**: this is the Modal-backed training entry point we used to
produce the VoxClinBench-Base reference baseline numbers in the
paper (Table~2 "VoxClinBench-Base" column). Submitters who want to
submit their OWN model to the leaderboard do NOT need to use this
code — they can run any training pipeline they like and submit a
two-column ``subject_id,predicted_prob`` CSV to ``voxbench eval``.

This file is kept in the release so that (a) our published per-seed
prediction CSVs are auditable against the exact code that produced
them, and (b) reviewers or future submitters who DO want to
reproduce / extend VoxClinBench-Base have the full training harness.
Running it requires a Modal credential and the credentialed upstream
corpus (PhysioNet for B2AI).

Multi-task disease classifier for the VoxClinBench Tier-2 panel.
7-branch CNN-Transformer architecture trained on Modal cloud GPUs.

This file is the Modal shell: it owns the app / image / volume declarations
and the two @app.function entry points (preprocess, train_model).
All reusable logic lives in the sibling packages:
  config.py          — constants, CONFIG dict, tier definitions
  data/features.py   — array reconstruction helpers
  data/labels.py     — label loading, participant splits, static features
  data/dataset.py    — VoiceDataset, normalization stats, DataLoaders
  model/branches.py  — BranchD/E/F/G + masked stats-pooling helpers
  model/classifier.py— VoiceDiseaseModel
  training/loss.py   — compute_pos_weights, make_loss_fn
  training/eval.py   — evaluate_auroc, EarlyStopping
  training/utils.py  — _setup_run_logging, _prune_matching_files

Run:
    # Preprocess only (CPU, ~30–45 min, idempotent):
    /opt/anaconda3/envs/11667/bin/modal run train.py::preprocess

    # Train a specific tier (GPU A10G), detached:
    /opt/anaconda3/envs/11667/bin/modal run -d train.py::main --tier 6
"""

import copy
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import modal

from voxbench.config import (
    MODALITY_KEYS, DISEASE_LIST, CORE_DISEASES,
    TIER_DISEASES, TIER_MODALITIES, CONFIG, N_STATIC_FEATURES, N_MODALITIES, TRAIN_PROFILES,
)
from voxbench.data.features import (
    pad_or_truncate,
    reconstruct_2d, reconstruct_ppg, reconstruct_ema, reconstruct_prosodic,
    compute_macro_prosodic, N_MACRO_PROS,
)
from voxbench.data.labels import (
    _pid_to_int, build_task_manifest, load_labels, make_splits,
    load_static_features, get_static_feature_cols,
)
from voxbench.data.dataset import VoiceDataset, compute_normalization_stats, make_dataloaders
from voxbench.model import VoiceDiseaseModel
from voxbench.training.loss import compute_pos_weights, make_loss_fn
from voxbench.training.eval import evaluate_auroc, EarlyStopping
from voxbench.training.utils import _setup_run_logging, _prune_matching_files


def _resolve_enabled_modalities(tier: int, enabled_modalities_csv: str = "") -> dict[str, bool]:
    enabled = dict(TIER_MODALITIES[tier])
    if not enabled_modalities_csv.strip():
        return enabled
    requested = {m.strip() for m in enabled_modalities_csv.split(",") if m.strip()}
    unknown = sorted(requested - set(MODALITY_KEYS))
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}")
    return {key: (key in requested) for key in MODALITY_KEYS}


def _modality_suffix(enabled: dict[str, bool]) -> str:
    return "mods_" + "_".join([key for key in MODALITY_KEYS if enabled.get(key, False)])


def _resolve_train_config(train_profile: str = "default") -> dict:
    cfg = copy.deepcopy(CONFIG)
    profile_key = train_profile.strip() or "default"
    if profile_key not in TRAIN_PROFILES:
        raise ValueError(f"Unknown train_profile='{profile_key}'. Expected one of {sorted(TRAIN_PROFILES)}")
    cfg.update(TRAIN_PROFILES[profile_key])
    return cfg


def _build_fair_subset(
    h5_path: str,
    norm_stats: dict,
    tier_enabled: dict[str, bool],
    disease_idx: list[int],
    eligible_pid_union: set[int],
):
    with h5py.File(h5_path, "r") as hf:
        meta = hf["metadata"]
        all_uids = [u.decode() for u in meta["uids"][:]]
        labels = meta["labels"][:]
        pids = meta["pids"][:].tolist()
        task_names = [t.decode() for t in meta["task_names"][:]]

    fair_idx = [i for i, pid in enumerate(pids) if pid in eligible_pid_union]
    fair_ds = VoiceDataset(
        h5_path=h5_path,
        uid_list=[all_uids[i] for i in fair_idx],
        labels=labels[fair_idx],
        norm_stats=norm_stats,
        enabled=tier_enabled,
        disease_idx=disease_idx,
        training=False,
        pids=[pids[i] for i in fair_idx],
        task_names=[task_names[i] for i in fair_idx],
    )
    return fair_ds


# =============================================================================
# Modal infrastructure
# =============================================================================

_WANDB_KEY = os.environ.get("WANDB_API_KEY", "")

app    = modal.App(os.environ.get("VOXBENCH_MODAL_APP", "voxclinbench-train"))
volume = modal.Volume.from_name(os.environ.get("VOXBENCH_MODAL_VOLUME", "voxclinbench-data"))
image  = modal.Image.from_registry(
    "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
).pip_install([
    "h5py>=3.10",
    "pyarrow>=14",
    "scikit-learn>=1.4",
    "pandas>=2.0",
    "numpy>=1.26",
    "tqdm",
    "torchvision>=0.20",
    "wandb>=0.17",
]).add_local_python_source("config", "data", "model", "training")


# =============================================================================
# SECTION 1: preprocess() — CPU Modal function
# =============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=8,
    memory=32768,
    timeout=7200,
)
def preprocess():
    """Stream all parquet files and write fixed-size tensors into HDF5.

    Always writes ALL modalities so any training tier can read from the same
    file without re-running preprocessing.  Skipped if a valid HDF5 exists.
    """
    import pyarrow.parquet as pq

    cfg      = CONFIG
    root     = cfg["data_root"]
    h5_path  = cfg["hdf5_path"]
    t_max    = cfg["T_MAX"]

    # ── Idempotency + concurrency check ──────────────────────────────────────
    import time

    lock_path = h5_path + ".lock"
    tmp_h5_path = h5_path + ".tmp"

    def _check_valid() -> bool:
        """Return True iff the HDF5 file exists and passes all schema checks."""
        if not os.path.exists(h5_path):
            return False
        try:
            with h5py.File(h5_path, "r") as _hf:
                _meta = _hf.get("metadata")
                ok = (
                    _meta is not None
                    and all(k in _meta for k in ("uids", "pids", "labels", "task_names"))
                    and "splits_json" in _meta.attrs
                    and json.loads(_meta.attrs.get("diseases", "[]")) == DISEASE_LIST
                )
                if ok:
                    ok = int(_meta.attrs.get("n_static_features", 0)) == N_STATIC_FEATURES
                if ok:
                    ok = int(_meta.attrs.get("n_modalities", 0)) == N_MODALITIES
            return ok
        except Exception:
            return False

    # Refresh volume view so we see updates from other containers.
    # Must happen before opening any files on the volume.
    volume.reload()
    run_log_path = _setup_run_logging("preprocess")

    # Fast path: already valid (common case when tiers run in parallel).
    if _check_valid():
        print(f"[preprocess] HDF5 valid at {h5_path} — skipping.")
        return

    # Acquire a lock atomically so only one container rebuilds the shared HDF5.
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as _lf:
                _lf.write(f"{os.getpid()}\n")
            volume.commit()
            break
        except FileExistsError:
            print("[preprocess] Lock file found — another container is preprocessing. Waiting …")
            time.sleep(30)
            volume.reload()
            if _check_valid():
                print(f"[preprocess] HDF5 valid at {h5_path} — skipping.")
                return

    # Remove any previous invalid outputs so we rebuild from a clean slate.
    if os.path.exists(h5_path):
        print("[preprocess] HDF5 schema changed — regenerating.")
        os.remove(h5_path)
    if os.path.exists(tmp_h5_path):
        os.remove(tmp_h5_path)

    feat_dir = os.path.join(root, "features")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # ── Inner helpers (parquet I/O, not reused outside preprocess) ───────────

    def _iter_key_cols(parquet_path: str, cols: list[str]) -> pd.DataFrame:
        pf = pq.ParquetFile(parquet_path)
        return pd.concat(
            [batch.to_pandas() for batch in pf.iter_batches(batch_size=512, columns=cols)],
            ignore_index=True,
        )

    def _key_set(df: pd.DataFrame) -> set:
        return set(zip(df["_pid_int"], df["session_id"], df["task_name"]))

    def _filter_to_keys(df: pd.DataFrame, ks: set) -> pd.DataFrame:
        mask = df.apply(
            lambda r: (r["_pid_int"], r["session_id"], r["task_name"]) in ks, axis=1
        )
        return df[mask].reset_index(drop=True)

    def _build_key_index(parquet_path: str) -> dict[tuple, int]:
        idx: dict[tuple, int] = {}
        row = 0
        for batch in pq.ParquetFile(parquet_path).iter_batches(
            batch_size=512, columns=["participant_id", "session_id", "task_name"]
        ):
            pdf = batch.to_pandas()
            for _, r in pdf.iterrows():
                idx[(_pid_to_int(r["participant_id"]), r["session_id"], r["task_name"])] = row
                row += 1
        return idx

    def _load_scalar_parquet(path: str, col: str) -> dict[tuple, np.ndarray]:
        result = {}
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for _, row in batch.to_pandas().iterrows():
                key = (_pid_to_int(row["participant_id"]), row["session_id"], row["task_name"])
                result[key] = np.asarray(row[col], dtype=np.float32).ravel()
        return result

    # ── Build recording index (inner-join across all modalities) ─────────────
    mod_file_map = {
        "spec": "torchaudio_spectrogram.parquet",
        "mel":  "torchaudio_mel_spectrogram.parquet",
        "mfcc": "torchaudio_mfcc.parquet",
        "ppg":  "ppgs.parquet",
        "ema":  "sparc_ema.parquet",
    }
    loud_path   = os.path.join(feat_dir, "sparc_loudness.parquet")
    period_path = os.path.join(feat_dir, "sparc_periodicity.parquet")
    pitch_path  = os.path.join(feat_dir, "sparc_pitch.parquet")

    print("[preprocess] Building recording index …")
    ref_df = _iter_key_cols(
        os.path.join(feat_dir, mod_file_map["spec"]),
        ["participant_id", "session_id", "task_name"],
    )
    ref_df["_pid_int"] = ref_df["participant_id"].apply(_pid_to_int)
    print(f"  Base (spec): {len(ref_df)} recordings")

    for mod, fname in mod_file_map.items():
        if mod == "spec":
            continue
        other = _iter_key_cols(
            os.path.join(feat_dir, fname), ["participant_id", "session_id", "task_name"]
        )
        other["_pid_int"] = other["participant_id"].apply(_pid_to_int)
        ref_df = _filter_to_keys(ref_df, _key_set(other))
        print(f"  After {mod}: {len(ref_df)} recordings")

    for path, label in [(loud_path, "loud"), (period_path, "period"), (pitch_path, "pitch")]:
        other = _iter_key_cols(path, ["participant_id", "session_id", "task_name"])
        other["_pid_int"] = other["participant_id"].apply(_pid_to_int)
        ref_df = _filter_to_keys(ref_df, _key_set(other))
        print(f"  After {label}: {len(ref_df)} recordings")

    static_df = load_static_features(root)  # participant_id already normalised to int
    static_feat_cols = get_static_feature_cols(static_df)
    ref_df = _filter_to_keys(
        ref_df,
        set(zip(static_df["participant_id"], static_df["session_id"], static_df["task_name"])),
    )
    print(f"  After static: {len(ref_df)} recordings")

    static_lookup = {
        (r["participant_id"], r["session_id"], r["task_name"]): r[static_feat_cols].values.astype(np.float32)
        for _, r in static_df.iterrows()
    }

    ref_df = ref_df.reset_index(drop=True)
    ref_df["uid"] = ref_df.index.astype(str)
    N = len(ref_df)
    print(f"[preprocess] Final count: {N} recordings")

    all_pids = ref_df["_pid_int"].tolist()
    all_uids = ref_df["uid"].tolist()

    # ── Labels + splits ───────────────────────────────────────────────────────
    labels_by_pid = load_labels(root)
    labels_matrix = np.zeros((N, len(DISEASE_LIST)), dtype=np.float32)
    for i, pid in enumerate(all_pids):
        if pid in labels_by_pid:
            labels_matrix[i] = labels_by_pid[pid]

    splits = make_splits(root, seed=cfg["seed"],
                         train_frac=cfg["train_frac"], val_frac=cfg["val_frac"])

    # ── Build parquet row-index maps ──────────────────────────────────────────
    print("[preprocess] Building key indices …")
    key_indices: dict[str, dict] = {}
    for mod, fname in mod_file_map.items():
        print(f"  {mod} …")
        key_indices[mod] = _build_key_index(os.path.join(feat_dir, fname))
    for label, path in [("loud", loud_path), ("period", period_path), ("pitch", pitch_path)]:
        print(f"  {label} …")
        key_indices[label] = _build_key_index(path)

    # ── Write HDF5 ────────────────────────────────────────────────────────────
    print(f"[preprocess] Writing HDF5 to {h5_path} …")
    silent = not os.isatty(1)

    def _write_mod(grp, key: str, arr: np.ndarray):
        grp.create_dataset(key, data=arr, dtype="float32",
                           compression="gzip", compression_opts=4)

    def _stream_write(mod: str, parquet_path: str, reconstruct_fn, batch_size: int = 64):
        pairs = sorted(
            ((key_indices[mod][(r["_pid_int"], r["session_id"], r["task_name"])], i)
             for i, r in ref_df.iterrows()
             if (r["_pid_int"], r["session_id"], r["task_name"]) in key_indices[mod]),
            key=lambda x: x[0],
        )
        if not pairs:
            return
        needed  = {p[0] for p in pairs}
        row2ref = {p[0]: p[1] for p in pairs}
        pf      = pq.ParquetFile(parquet_path)
        global_r = 0
        for batch in tqdm(pf.iter_batches(batch_size=batch_size),
                          desc=mod, leave=False, disable=silent):
            pdf = batch.to_pandas()
            for local_i, (_, row) in enumerate(pdf.iterrows()):
                gidx = global_r + local_i
                if gidx in needed:
                    ref_i = row2ref[gidx]
                    arr   = pad_or_truncate(reconstruct_fn(row), t_max)
                    _write_mod(recs_grp[all_uids[ref_i]], mod, arr)
            global_r += len(pdf)

    with h5py.File(tmp_h5_path, "w") as hf:
        recs_grp = hf.require_group("recordings")
        for uid in all_uids:
            recs_grp.require_group(uid)

        # 2-D modalities (large files, streamed)
        _stream_write("spec", os.path.join(feat_dir, mod_file_map["spec"]),
                      lambda r: reconstruct_2d(r, "spectrogram"))
        _stream_write("mel",  os.path.join(feat_dir, mod_file_map["mel"]),
                      lambda r: reconstruct_2d(r, "mel_spectrogram"))
        _stream_write("mfcc", os.path.join(feat_dir, mod_file_map["mfcc"]),
                      lambda r: reconstruct_2d(r, "mfcc"))
        _stream_write("ppg",  os.path.join(feat_dir, mod_file_map["ppg"]),
                      reconstruct_ppg)
        _stream_write("ema",  os.path.join(feat_dir, mod_file_map["ema"]),
                      reconstruct_ema, batch_size=512)

        # Prosodic (small; load into RAM first)
        print("[preprocess] Loading prosodic files into RAM …")
        loud_d   = _load_scalar_parquet(loud_path,   "loudness")
        period_d = _load_scalar_parquet(period_path, "periodicity")
        pitch_d  = _load_scalar_parquet(pitch_path,  "pitch")

        print("[preprocess] Writing prosodic …")
        for i, r in tqdm(ref_df.iterrows(), total=N, desc="pros", leave=False, disable=silent):
            key = (r["_pid_int"], r["session_id"], r["task_name"])
            if key in loud_d and key in period_d and key in pitch_d:
                mat = reconstruct_prosodic(loud_d[key], period_d[key], pitch_d[key])
                _write_mod(recs_grp[all_uids[i]], "pros", pad_or_truncate(mat, t_max))

        # Static + macro prosodic
        # Each recording's static vector = [131 openSMILE/Praat | 6 macro prosodic]
        # Macro prosodic features are derived from the per-frame loudness/periodicity
        # signals already in RAM, so no extra I/O is needed here.
        print("[preprocess] Writing static + macro prosodic features …")
        _zeros_macro = np.zeros(N_MACRO_PROS, dtype=np.float32)
        for i, r in tqdm(ref_df.iterrows(), total=N, desc="static", leave=False, disable=silent):
            key = (r["_pid_int"], r["session_id"], r["task_name"])
            if key in static_lookup:
                macro = (
                    compute_macro_prosodic(loud_d[key], period_d[key], pitch_d[key])
                    if (key in loud_d and key in period_d and key in pitch_d)
                    else _zeros_macro
                )
                _write_mod(
                    recs_grp[all_uids[i]], "static",
                    np.concatenate([static_lookup[key], macro]),
                )

        # Available mask
        print("[preprocess] Writing available_mask …")
        for uid in all_uids:
            grp  = recs_grp[uid]
            mask = np.array([k in grp for k in MODALITY_KEYS], dtype=bool)
            grp.create_dataset("available_mask", data=mask)

        # Metadata
        print("[preprocess] Writing metadata …")
        meta = hf.require_group("metadata")
        meta.create_dataset("uids",       data=np.array([u.encode() for u in all_uids]))
        meta.create_dataset("pids",       data=np.array(all_pids, dtype=np.int64))
        meta.create_dataset("task_names", data=np.array([str(t).encode()
                                                          for t in ref_df["task_name"]]))
        meta.create_dataset("labels",     data=labels_matrix)
        meta.attrs["splits_json"]       = json.dumps(splits)
        meta.attrs["diseases"]          = json.dumps(DISEASE_LIST)
        meta.attrs["n_static_features"] = N_STATIC_FEATURES
        meta.attrs["n_modalities"]      = N_MODALITIES

    os.replace(tmp_h5_path, h5_path)
    volume.commit()

    # Release lock so any waiting peers can re-validate and skip.
    if os.path.exists(lock_path):
        os.remove(lock_path)
    volume.commit()
    print("[preprocess] Done — HDF5 committed.")

    latest_ptr = "/data/checkpoints/run_logs/latest_preprocess_log.txt"
    os.makedirs(os.path.dirname(latest_ptr), exist_ok=True)
    with open(latest_ptr, "w") as f:
        f.write(run_log_path + "\n")
    volume.commit()


# =============================================================================
# SECTION 2: train_model() — GPU Modal function
# =============================================================================

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    memory=16384,
    timeout=43200,
    secrets=[modal.Secret.from_dict({"WANDB_API_KEY": _WANDB_KEY})] if _WANDB_KEY else [],
)
def train_model(
    tier: int = 5,
    fair_eval: bool = False,
    enabled_modalities_csv: str = "",
    run_name_suffix: str = "",
    train_profile: str = "default",
):
    """GPU training: load HDF5, train model for the given tier, save checkpoint.

    Runs a quick local HDF5 validity check first.  Only spawns preprocess() if
    the file is missing or its schema doesn't match the current config.  This
    avoids an unnecessary CPU container cold-start when the HDF5 already exists.
    """
    import wandb as _wandb

    cfg     = _resolve_train_config(train_profile)
    h5_path = cfg["hdf5_path"]

    # ── Quick HDF5 validity check (runs inside the GPU container, no extra spawn)
    volume.reload()

    def _h5_valid() -> bool:
        if not os.path.exists(h5_path):
            return False
        try:
            with h5py.File(h5_path, "r") as hf:
                meta = hf.get("metadata")
                ok = (
                    meta is not None
                    and all(k in meta for k in ("uids", "pids", "labels", "task_names"))
                    and "splits_json" in meta.attrs
                    and json.loads(meta.attrs.get("diseases", "[]")) == DISEASE_LIST
                    and int(meta.attrs.get("n_static_features", 0)) == N_STATIC_FEATURES
                    and int(meta.attrs.get("n_modalities", 0)) == N_MODALITIES
                )
            return ok
        except Exception:
            return False

    if _h5_valid():
        print(f"[train_model] HDF5 valid at {h5_path} — skipping preprocess.")
    else:
        print("[train_model] HDF5 missing or invalid — running preprocess …")
        preprocess.remote()
        print("[train_model] Preprocess done — starting training.")
    tier_diseases = TIER_DISEASES[tier]
    tier_enabled  = _resolve_enabled_modalities(tier, enabled_modalities_csv)
    n_diseases    = len(tier_diseases)
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_filter_map_val = None
    patient_filter_map_test = None

    run_suffix_parts = []
    if fair_eval:
        run_suffix_parts.append("fair_eval")
    if enabled_modalities_csv.strip():
        run_suffix_parts.append(_modality_suffix(tier_enabled))
    if run_name_suffix.strip():
        run_suffix_parts.append(run_name_suffix.strip().replace(" ", "_"))
    if train_profile.strip() and train_profile.strip() != "default":
        run_suffix_parts.append(train_profile.strip())
    run_suffix = f"_{'_'.join(run_suffix_parts)}" if run_suffix_parts else ""
    tier_dir     = os.path.join(cfg["ckpt_dir"], f"tier{tier}{run_suffix}")
    run_logs_dir = os.path.join(tier_dir, "run_logs")
    code_dir     = os.path.join(tier_dir, "code_snapshot")
    for d in (tier_dir, run_logs_dir, code_dir):
        os.makedirs(d, exist_ok=True)

    run_log_path = _setup_run_logging("train", tier=tier, base_dir=run_logs_dir)
    ckpt_path    = os.path.join(tier_dir, "best.pt")
    resume_path  = os.path.join(tier_dir, "resume.pt")

    # ── Code snapshot (all source modules) ───────────────────────────────────
    snap_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = None
    try:
        src_root = os.path.dirname(os.path.abspath(__file__))
        for item in ["train.py", "config.py", "data", "model", "training"]:
            src = os.path.join(src_root, item)
            dst = os.path.join(code_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        cfg_path = os.path.join(code_dir, f"config_tier{tier}_{snap_ts}.json")
        with open(cfg_path, "w") as f:
            json.dump(
                {
                    "tier": tier,
                    "config": cfg,
                    "diseases": tier_diseases,
                    "enabled_modalities": tier_enabled,
                    "run_name_suffix": run_name_suffix,
                    "train_profile": train_profile,
                },
                f,
                indent=2,
            )
        volume.commit()
        print(f"[train_model] Code snapshot saved to {code_dir}")
    except Exception as e:
        print(f"[train_model] Warning: code snapshot failed: {e}")

    print(f"[train_model] Tier {tier} | {n_diseases} diseases | device: {device}")
    print(f"  Diseases:   {tier_diseases}")
    print(f"  Modalities: {[k for k, v in tier_enabled.items() if v]}")
    print(f"  Fair eval:  {fair_eval}")
    print(f"  Profile:    {train_profile}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = bool(os.environ.get("WANDB_API_KEY", ""))
    if use_wandb:
        _wandb.init(
            project=os.environ.get("VOXBENCH_WANDB_PROJECT", "voxclinbench"),
            name=f"tier{tier}{run_suffix}-{snap_ts}",
            config={
                **cfg,
                "tier":       tier,
                "diseases":   tier_diseases,
                "modalities": [k for k, v in tier_enabled.items() if v],
                "n_diseases": n_diseases,
            },
            tags=[f"tier{tier}"],
        )
        print("[train_model] WandB initialized.")
    else:
        print("[train_model] WandB not configured — logging to stdout only.")

    # ── Load HDF5 metadata ───────────────────────────────────────────────────
    with h5py.File(h5_path, "r") as hf:
        meta            = hf["metadata"]
        pids            = meta["pids"][:].tolist()
        labels          = meta["labels"][:]
        splits          = json.loads(meta.attrs["splits_json"])
        stored_diseases = json.loads(meta.attrs.get("diseases", "[]"))

    if fair_eval:
        manifest = build_task_manifest(
            cfg["data_root"],
            tier_diseases,
            seed=cfg["seed"],
            val_frac_within_train=0.25,
        )
        patient_filter_map_val = {
            disease: set(manifest["tasks"][disease]["positive"]["val"]) | set(manifest["tasks"][disease]["negative"]["val"])
            for disease in tier_diseases
        }
        patient_filter_map_test = {
            disease: set(manifest["tasks"][disease]["positive"]["test"]) | set(manifest["tasks"][disease]["negative"]["test"])
            for disease in tier_diseases
        }

    if not stored_diseases:
        # Backward compat: HDF5 built before disease names were stored
        if labels.shape[1] == len(CORE_DISEASES):
            stored_diseases = CORE_DISEASES
        elif labels.shape[1] == len(DISEASE_LIST):
            stored_diseases = DISEASE_LIST
        else:
            raise ValueError(
                f"HDF5 has {labels.shape[1]} label columns but no disease metadata. "
                "Please re-run preprocess."
            )
    if len(stored_diseases) != labels.shape[1]:
        raise ValueError(
            f"HDF5 disease metadata ({len(stored_diseases)}) ≠ label columns "
            f"({labels.shape[1]}). Re-run preprocess."
        )
    missing = [d for d in tier_diseases if d not in stored_diseases]
    if missing:
        raise ValueError(
            f"Tier {tier} diseases not in HDF5 label space: {missing}. Re-run preprocess."
        )

    disease_idx = [stored_diseases.index(d) for d in tier_diseases]
    print(f"[train_model] Label space in HDF5: {len(stored_diseases)} diseases")

    train_pids = set(splits["train"])
    val_pids   = set(splits["val"])
    test_pids  = set(splits["test"])
    train_idx  = [i for i, p in enumerate(pids) if p in train_pids]
    val_idx    = [i for i, p in enumerate(pids) if p in val_pids]
    test_idx   = [i for i, p in enumerate(pids) if p in test_pids]

    with h5py.File(h5_path, "r") as hf:
        all_uids = [u.decode() for u in hf["metadata"]["uids"][:]]
    train_uids = [all_uids[i] for i in train_idx]

    # ── Normalization stats ───────────────────────────────────────────────────
    enabled_mods    = [k for k in MODALITY_KEYS if tier_enabled.get(k, False)]
    norm_cache_attr = f"norm_stats_mods__{'_'.join(enabled_mods)}"

    def _to_np_norm(obj: dict) -> dict[str, dict[str, np.ndarray]]:
        return {k: {"mean": np.asarray(v["mean"], dtype=np.float32),
                    "std":  np.asarray(v["std"],  dtype=np.float32)}
                for k, v in obj.items()}

    norm_stats_json = None
    if cfg.get("use_cached_norm_stats", False):
        with h5py.File(h5_path, "r") as hf:
            attr = hf["metadata"].attrs.get(norm_cache_attr)
            if attr:
                norm_stats_json = json.loads(attr)

    if norm_stats_json is not None:
        norm_stats = _to_np_norm(norm_stats_json)
        print(f"[train_model] Loaded cached normalization stats ({norm_cache_attr})")
    else:
        print("[train_model] Computing normalization stats …")
        norm_stats      = compute_normalization_stats(h5_path, train_uids, tier_enabled)
        norm_stats_json = {k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()}
                           for k, v in norm_stats.items()}
        with h5py.File(h5_path, "a") as hf:
            hf["metadata"].attrs[norm_cache_attr]             = json.dumps(norm_stats_json)
            hf["metadata"].attrs[f"norm_stats_tier{tier}"]   = json.dumps(norm_stats_json)
        volume.commit()

    print("[train_model] Normalization stats summary:")
    for mod in sorted(norm_stats.keys()):
        m, s = norm_stats[mod]["mean"], norm_stats[mod]["std"]
        print(f"  {mod}: mean∈[{m.min():.4f}, {m.max():.4f}]  "
              f"std∈[{s.min():.4f}, {s.max():.4f}]")

    # ── Positive weights + split labels ──────────────────────────────────────
    train_labels_tier = labels[train_idx][:, disease_idx]
    val_labels_tier   = labels[val_idx][:,   disease_idx]
    test_labels_tier  = labels[test_idx][:,  disease_idx]

    print("[train_model] Positives by split (recording-level):")
    for i, dname in enumerate(tier_diseases):
        print(f"  {dname}: train={int(train_labels_tier[:, i].sum())}  "
              f"val={int(val_labels_tier[:, i].sum())}  "
              f"test={int(test_labels_tier[:, i].sum())}")

    pos_w = compute_pos_weights(
        train_labels_tier, cfg["pos_weight_min"], cfg["pos_weight_max"]
    ).to(device)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, _ = make_dataloaders(
        h5_path, norm_stats, tier_enabled, disease_idx,
        batch_size=cfg["batch_size"], n_workers=cfg["n_workers"],
        pos_weights=pos_w.cpu(),
    )
    fair_val_loader = None
    fair_test_loader = None
    if fair_eval:
        fair_val_union = set().union(*patient_filter_map_val.values())
        fair_test_union = set().union(*patient_filter_map_test.values())
        fair_val_ds = _build_fair_subset(
            h5_path=h5_path,
            norm_stats=norm_stats,
            tier_enabled=tier_enabled,
            disease_idx=disease_idx,
            eligible_pid_union=fair_val_union,
        )
        fair_test_ds = _build_fair_subset(
            h5_path=h5_path,
            norm_stats=norm_stats,
            tier_enabled=tier_enabled,
            disease_idx=disease_idx,
            eligible_pid_union=fair_test_union,
        )
        fair_val_loader = DataLoader(
            fair_val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["n_workers"],
            pin_memory=True,
            persistent_workers=(cfg["n_workers"] > 0),
        )
        fair_test_loader = DataLoader(
            fair_test_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["n_workers"],
            pin_memory=True,
            persistent_workers=(cfg["n_workers"] > 0),
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[train_model] Building model …")
    model = VoiceDiseaseModel(
        n_diseases=n_diseases,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        n_layers=cfg["n_transformer_layers"],
        dropout_fusion=cfg["dropout_fusion"],
        dropout_head=cfg["dropout_head"],
        head_hidden_dim=cfg.get("head_hidden_dim", 256),
        modality_dropout_prob=cfg.get("modality_dropout_prob", 0.0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")
    if use_wandb:
        _wandb.config.update({"n_params": n_params})

    # ── EMA model ─────────────────────────────────────────────────────────────
    ema_decay   = float(cfg.get("ema_decay", 0.0))
    ema_enabled = ema_decay > 0.0
    ema_model   = None
    if ema_enabled:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"  EMA model initialized (decay={ema_decay})")
    else:
        print("  EMA disabled (ema_decay = 0)")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    backbone_ids    = {id(p) for p in list(model.branch_a.parameters())
                                   + list(model.branch_b.parameters())
                                   + list(model.branch_c.parameters())}
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    new_params      = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["lr_backbone"]},
        {"params": new_params,      "lr": cfg["lr_new"]},
    ], weight_decay=cfg["weight_decay"])

    warmup_epochs = int(cfg["lr_warmup_epochs"])
    cosine_t_max  = int(cfg["cosine_t_max"])
    base_lrs      = [pg["lr"] for pg in optimizer.param_groups]
    min_ratios    = [min(1.0, 1e-6 / max(lr, 1e-12)) for lr in base_lrs]

    def _lr_lambda(min_ratio: float):
        def _fn(step: int) -> float:
            e = step + 1
            if e <= warmup_epochs:
                return 0.1 + 0.9 * (e / max(warmup_epochs, 1))
            t   = min(max(e - warmup_epochs, 0), cosine_t_max)
            cos = 0.5 * (1.0 + math.cos(math.pi * t / max(cosine_t_max, 1)))
            return min_ratio + (1.0 - min_ratio) * cos
        return _fn

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[_lr_lambda(r) for r in min_ratios]
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = make_loss_fn(
        pos_weights=pos_w,
        label_smoothing=cfg["label_smoothing"],
        focal_gamma=cfg.get("focal_gamma", 0.0),
    )

    # ── Misc training state ───────────────────────────────────────────────────
    scaler      = torch.amp.GradScaler("cuda", enabled=False)
    es          = EarlyStopping(patience=cfg["patience"])
    best_val    = -float("inf")
    accum_steps = cfg.get("accumulation_steps", 1)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    amp_enabled = bool(cfg.get("amp_enabled", False))
    start_epoch = 1

    # ── Optional checkpoint resume ────────────────────────────────────────────
    if cfg.get("auto_resume", False) and os.path.exists(resume_path):
        print(f"[train_model] Resuming from {resume_path} …")
        ckpt_r = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r["state_dict"])
        optimizer.load_state_dict(ckpt_r["optimizer"])
        scheduler.load_state_dict(ckpt_r["scheduler"])
        best_val    = ckpt_r["best_val"]
        es.best     = ckpt_r["es_best"]
        es.counter  = ckpt_r["es_counter"]
        start_epoch = ckpt_r["epoch"] + 1
        if ema_enabled and ema_model is not None:
            key = "ema_state_dict" if "ema_state_dict" in ckpt_r else "state_dict"
            ema_model.load_state_dict(ckpt_r[key])
        print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    def _tensor_health(name: str, t: torch.Tensor):
        x = t.float()
        print(
            f"  [{name}] shape={tuple(x.shape)} "
            f"nan={int(torch.isnan(x).sum())} "
            f"inf={int(torch.isinf(x).sum())} "
            f"min={float(torch.nan_to_num(x).min()):.4f} "
            f"max={float(torch.nan_to_num(x).max()):.4f}"
        )

    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        use_mixup  = (mixup_alpha > 0) and (epoch > cfg["lr_warmup_epochs"])
        optimizer.zero_grad(set_to_none=True)
        epoch_start_time = time.perf_counter()
        data_wait_time = 0.0
        forward_time = 0.0
        backward_optim_time = 0.0
        train_samples = 0
        batch_fetch_start = time.perf_counter()

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Ep{epoch}", leave=False, disable=not os.isatty(1))
        ):
            data_wait_time += time.perf_counter() - batch_fetch_start
            train_samples += int(batch["label"].size(0))
            # MixUp (applied after warmup)
            if use_mixup:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(batch["label"].size(0))
                for key in ("spec", "mel", "mfcc", "ppg", "ema", "pros", "static"):
                    batch[key] = lam * batch[key] + (1 - lam) * batch[key][idx]
                batch["label"]     = lam * batch["label"]     + (1 - lam) * batch["label"][idx]
                batch["available"] = batch["available"] | batch["available"][idx]
                for key in ("act_len_ppg", "act_len_ema", "act_len_pros"):
                    batch[key] = torch.max(batch[key], batch[key][idx])

            forward_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.bfloat16):
                logits = model(
                    batch["spec"].to(device),
                    batch["mfcc"].to(device),    batch["mel"].to(device),
                    batch["ppg"].to(device),     batch["ema"].to(device),
                    batch["pros"].to(device),    batch["static"].to(device),
                    batch["available"].to(device),
                    act_len_ppg=batch["act_len_ppg"].to(device),
                    act_len_ema=batch["act_len_ema"].to(device),
                    act_len_pros=batch["act_len_pros"].to(device),
                )
                if not torch.isfinite(logits).all():
                    print(f"[warn] Non-finite logits at epoch={epoch} step={step}; skipping.")
                    _tensor_health("logits", logits.detach())
                    for key in ("spec", "mel", "mfcc", "ppg", "ema", "pros", "static"):
                        _tensor_health(key, batch[key])
                    optimizer.zero_grad(set_to_none=True)
                    batch_fetch_start = time.perf_counter()
                    continue

                loss = loss_fn(logits, batch["label"].to(device)) / accum_steps
            forward_time += time.perf_counter() - forward_start

            if not torch.isfinite(loss):
                print(f"[warn] Non-finite loss at epoch={epoch} step={step}; skipping.")
                _tensor_health("logits", logits.detach())
                optimizer.zero_grad(set_to_none=True)
                batch_fetch_start = time.perf_counter()
                continue

            backward_start = time.perf_counter()
            scaler.scale(loss).backward()

            is_last = (step + 1 == len(train_loader))
            if (step + 1) % accum_steps == 0 or is_last:
                scaler.unscale_(optimizer)
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    print(f"[warn] Non-finite gradients at epoch={epoch} step={step}; skipping.")
                    optimizer.zero_grad(set_to_none=True)
                    batch_fetch_start = time.perf_counter()
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if ema_enabled and ema_model is not None:
                    with torch.no_grad():
                        for p_ema, p_train in zip(ema_model.parameters(), model.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p_train.data, alpha=1.0 - ema_decay)
                        for b_ema, b_train in zip(ema_model.buffers(), model.buffers()):
                            b_ema.data.copy_(b_train.data)
            backward_optim_time += time.perf_counter() - backward_start

            total_loss += loss.item() * accum_steps
            n_batches  += 1
            batch_fetch_start = time.perf_counter()

        scheduler.step()
        avg_loss   = total_loss / max(n_batches, 1)
        eval_model = ema_model if (ema_enabled and ema_model is not None) else model
        val_start = time.perf_counter()
        val_macro, val_per = evaluate_auroc(
            eval_model, fair_val_loader if fair_eval else val_loader, device, tier_diseases,
            task_filter_keywords=cfg.get("task_filter_keywords", {}),
            patient_filter_map=patient_filter_map_val,
            min_support_patients=(1 if fair_eval else cfg.get("es_min_val_patients", 0)),
        )
        val_time = time.perf_counter() - val_start
        epoch_wall = time.perf_counter() - epoch_start_time
        train_samples_per_sec = train_samples / max(data_wait_time + forward_time + backward_optim_time, 1e-6)
        val_dataset_len = len((fair_val_loader.dataset if fair_eval else val_loader.dataset))
        val_samples_per_sec = val_dataset_len / max(val_time, 1e-6)

        log_dict: dict = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_macro_auroc": val_macro,
            "timing/data_wait_sec": data_wait_time,
            "timing/forward_sec": forward_time,
            "timing/backward_optim_sec": backward_optim_time,
            "timing/validation_sec": val_time,
            "timing/epoch_wall_sec": epoch_wall,
            "throughput/train_samples_per_sec": train_samples_per_sec,
            "throughput/val_samples_per_sec": val_samples_per_sec,
        }
        for name, auroc in zip(tier_diseases, val_per):
            if not math.isnan(auroc):
                log_dict[f"val_auroc/{name}"] = auroc

        print(f"Ep {epoch:3d} | loss={avg_loss:.4f} | val_macro={val_macro:.4f} "
              f"| best={best_val:.4f}")
        print(
            "  timing: "
            f"wait={data_wait_time:.1f}s "
            f"fwd={forward_time:.1f}s "
            f"bwd+opt={backward_optim_time:.1f}s "
            f"val={val_time:.1f}s "
            f"epoch={epoch_wall:.1f}s "
            f"train_sps={train_samples_per_sec:.1f} "
            f"val_sps={val_samples_per_sec:.1f}"
        )
        print("  val: " + "  ".join(
            f"{n}={('nan' if math.isnan(a) else f'{a:.4f}')}"
            for n, a in zip(tier_diseases, val_per)
        ))

        if use_wandb:
            _wandb.log(log_dict)

        if val_macro > best_val:
            best_val = val_macro
            torch.save({
                "epoch":      epoch,
                "tier":       tier,
                "state_dict": eval_model.state_dict(),
                "val_auroc":  val_macro,
                "norm_stats": norm_stats_json,
                "config":     cfg,
            }, ckpt_path)
            volume.commit()
            print(f"  → best checkpoint saved (val={val_macro:.4f})")

        torch.save({
            "epoch":          epoch,
            "tier":           tier,
            "state_dict":     model.state_dict(),
            "ema_state_dict": (ema_model.state_dict()
                               if (ema_enabled and ema_model is not None)
                               else model.state_dict()),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_val":       best_val,
            "es_best":        es.best,
            "es_counter":     es.counter,
        }, resume_path)
        volume.commit()

        if es.step(val_macro):
            print(f"[train_model] Early stopping at epoch {epoch}.")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("[train_model] Loading best checkpoint for test evaluation …")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test_macro, test_per = evaluate_auroc(
        model, fair_test_loader if fair_eval else test_loader, device, tier_diseases,
        task_filter_keywords=cfg.get("task_filter_keywords", {}),
        patient_filter_map=patient_filter_map_test,
    )

    print(f"\n{'='*50}")
    print(f"Tier {tier} Test Results")
    print(f"{'='*50}")
    print(f"  Macro AUROC: {test_macro:.4f}")
    print(f"  {'Disease':<30} {'AUROC':>6}")
    print(f"  {'-'*38}")
    for name, auroc in zip(tier_diseases, test_per):
        print(f"  {name:<30} {(f'{auroc:.4f}' if not math.isnan(auroc) else '   N/A'):>6}")

    if use_wandb:
        test_log = {"test_macro_auroc": test_macro}
        for name, auroc in zip(tier_diseases, test_per):
            if not math.isnan(auroc):
                test_log[f"test_auroc/{name}"] = auroc
        _wandb.log(test_log)
        _wandb.finish()

    # ── Cleanup old run logs (keep only the latest) ───────────────────────────
    latest_ptr = os.path.join(run_logs_dir, "latest_train_log.txt")
    with open(latest_ptr, "w") as f:
        f.write(run_log_path + "\n")

    removed_logs = _prune_matching_files(
        run_logs_dir,
        keep_basenames={os.path.basename(run_log_path)},
        matcher=lambda n: n.startswith(f"train_tier{tier}_") and n.endswith(".log"),
    )
    print(f"[cleanup] Removed {removed_logs} old run log(s).")
    volume.commit()
    print("[train_model] Done.")

    return {
        "tier":             tier,
        "test_macro_auroc": test_macro,
        "test_per_disease": {n: a for n, a in zip(tier_diseases, test_per)},
        "best_val_auroc":   best_val,
        "run_log_path":     run_log_path,
        "tier_dir":         tier_dir,
        "best_model_path":  ckpt_path,
        "fair_eval":        fair_eval,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    memory=16384,
    timeout=43200,
)
def eval_checkpoint(
    tier: int = 1,
    trained_with_fair_eval: bool = True,
    eval_fair: bool = False,
    enabled_modalities_csv: str = "",
    run_name_suffix: str = "",
    train_profile: str = "default",
):
    """Evaluate an existing checkpoint without retraining.

    Useful for:
    - re-evaluating a fair-eval checkpoint under the original unified protocol
    - recovering final test metrics when a training run finished awkwardly
    """
    assert tier in TIER_DISEASES, f"tier must be one of {sorted(TIER_DISEASES.keys())}"
    cfg = _resolve_train_config(train_profile)
    h5_path = cfg["hdf5_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tier_diseases = TIER_DISEASES[tier]
    tier_enabled = _resolve_enabled_modalities(tier, enabled_modalities_csv)
    run_suffix_parts = []
    if trained_with_fair_eval:
        run_suffix_parts.append("fair_eval")
    if enabled_modalities_csv.strip():
        run_suffix_parts.append(_modality_suffix(tier_enabled))
    if run_name_suffix.strip():
        run_suffix_parts.append(run_name_suffix.strip().replace(" ", "_"))
    if train_profile.strip() and train_profile.strip() != "default":
        run_suffix_parts.append(train_profile.strip())
    tier_dir = os.path.join(cfg["ckpt_dir"], f"tier{tier}{'_' + '_'.join(run_suffix_parts) if run_suffix_parts else ''}")
    ckpt_path = os.path.join(tier_dir, "best.pt")

    print(f"[eval_checkpoint] tier={tier}")
    print(f"[eval_checkpoint] trained_with_fair_eval={trained_with_fair_eval}")
    print(f"[eval_checkpoint] eval_fair={eval_fair}")
    print(f"[eval_checkpoint] ckpt_path={ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm_stats_json = ckpt["norm_stats"]
    norm_stats = {
        mod: {
            "mean": np.array(stats["mean"], dtype=np.float32),
            "std": np.array(stats["std"], dtype=np.float32),
        }
        for mod, stats in norm_stats_json.items()
    }

    with h5py.File(h5_path, "r") as hf:
        meta = hf["metadata"]
        stored_diseases = json.loads(meta.attrs["diseases"])

    disease_idx = [stored_diseases.index(d) for d in tier_diseases]
    _, _, test_loader, _ = make_dataloaders(
        h5_path, norm_stats, tier_enabled, disease_idx,
        batch_size=cfg["batch_size"], n_workers=cfg["n_workers"],
        pos_weights=None,
    )

    patient_filter_map = None
    eval_loader = test_loader
    if eval_fair:
        task_manifest = build_task_manifest(
            data_root=cfg["data_root"],
            task_names=tier_diseases,
            seed=cfg["seed"],
            val_frac_within_train=0.25,
        )
        patient_filter_map = {
            name: set(task_manifest["tasks"][name]["positive"]["test"])
            | set(task_manifest["tasks"][name]["negative"]["test"])
            for name in tier_diseases
        }
        eval_ds = _build_fair_subset(
            h5_path=h5_path,
            norm_stats=norm_stats,
            tier_enabled=tier_enabled,
            disease_idx=disease_idx,
            eligible_pid_union=set().union(*patient_filter_map.values()),
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["n_workers"],
            pin_memory=True,
            persistent_workers=(cfg["n_workers"] > 0),
        )

    model = VoiceDiseaseModel(
        n_diseases=len(tier_diseases),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        n_layers=cfg["n_transformer_layers"],
        dropout_fusion=cfg["dropout_fusion"],
        dropout_head=cfg["dropout_head"],
        head_hidden_dim=cfg.get("head_hidden_dim", 256),
        modality_dropout_prob=0.0,
    ).to(device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)

    test_macro, test_per = evaluate_auroc(
        model, eval_loader, device, tier_diseases,
        task_filter_keywords=cfg.get("task_filter_keywords", {}),
        patient_filter_map=patient_filter_map,
    )

    print(f"\n{'='*50}")
    print(f"Checkpoint Eval — Tier {tier}")
    print(f"{'='*50}")
    print(f"  Trained with fair_eval: {trained_with_fair_eval}")
    print(f"  Evaluated with fair_eval: {eval_fair}")
    print(f"  Macro AUROC: {test_macro:.4f}")
    print(f"  {'Disease':<30} {'AUROC':>6}")
    print(f"  {'-'*38}")
    for name, auroc in zip(tier_diseases, test_per):
        print(f"  {name:<30} {(f'{auroc:.4f}' if not math.isnan(auroc) else '   N/A'):>6}")

    return {
        "tier": tier,
        "trained_with_fair_eval": trained_with_fair_eval,
        "evaluated_with_fair_eval": eval_fair,
        "test_macro_auroc": test_macro,
        "test_per_disease": {n: a for n, a in zip(tier_diseases, test_per)},
        "checkpoint_path": ckpt_path,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    memory=16384,
    timeout=43200,
)
def export_checkpoint_predictions(
    tier: int = 1,
    trained_with_fair_eval: bool = True,
    eval_fair: bool = False,
    enabled_modalities_csv: str = "",
    run_name_suffix: str = "",
    train_profile: str = "default",
):
    """Export patient-level predictions from an existing checkpoint."""
    import pandas as pd

    assert tier in TIER_DISEASES, f"tier must be one of {sorted(TIER_DISEASES.keys())}"
    cfg = _resolve_train_config(train_profile)
    h5_path = cfg["hdf5_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tier_diseases = TIER_DISEASES[tier]
    tier_enabled = _resolve_enabled_modalities(tier, enabled_modalities_csv)
    run_suffix_parts = []
    if trained_with_fair_eval:
        run_suffix_parts.append("fair_eval")
    if enabled_modalities_csv.strip():
        run_suffix_parts.append(_modality_suffix(tier_enabled))
    if run_name_suffix.strip():
        run_suffix_parts.append(run_name_suffix.strip().replace(" ", "_"))
    if train_profile.strip() and train_profile.strip() != "default":
        run_suffix_parts.append(train_profile.strip())
    tier_dir = os.path.join(cfg["ckpt_dir"], f"tier{tier}{'_' + '_'.join(run_suffix_parts) if run_suffix_parts else ''}")
    ckpt_path = os.path.join(tier_dir, "best.pt")
    analysis_dir = os.path.join(tier_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm_stats = {
        mod: {
            "mean": np.array(stats["mean"], dtype=np.float32),
            "std": np.array(stats["std"], dtype=np.float32),
        }
        for mod, stats in ckpt["norm_stats"].items()
    }

    with h5py.File(h5_path, "r") as hf:
        meta = hf["metadata"]
        stored_diseases = json.loads(meta.attrs["diseases"])

    disease_idx = [stored_diseases.index(d) for d in tier_diseases]
    _, _, test_loader, _ = make_dataloaders(
        h5_path, norm_stats, tier_enabled, disease_idx,
        batch_size=cfg["batch_size"], n_workers=cfg["n_workers"],
        pos_weights=None,
    )

    patient_filter_map = None
    eval_loader = test_loader
    if eval_fair:
        task_manifest = build_task_manifest(
            data_root=cfg["data_root"],
            task_names=tier_diseases,
            seed=cfg["seed"],
            val_frac_within_train=0.25,
        )
        patient_filter_map = {
            name: set(task_manifest["tasks"][name]["positive"]["test"])
            | set(task_manifest["tasks"][name]["negative"]["test"])
            for name in tier_diseases
        }
        eval_ds = _build_fair_subset(
            h5_path=h5_path,
            norm_stats=norm_stats,
            tier_enabled=tier_enabled,
            disease_idx=disease_idx,
            eligible_pid_union=set().union(*patient_filter_map.values()),
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["n_workers"],
            pin_memory=True,
            persistent_workers=(cfg["n_workers"] > 0),
        )

    model = VoiceDiseaseModel(
        n_diseases=len(tier_diseases),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        n_layers=cfg["n_transformer_layers"],
        dropout_fusion=cfg["dropout_fusion"],
        dropout_head=cfg["dropout_head"],
        head_hidden_dim=cfg.get("head_hidden_dim", 256),
        modality_dropout_prob=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_logits, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            logits = model(
                batch["spec"].to(device),
                batch["mfcc"].to(device),    batch["mel"].to(device),
                batch["ppg"].to(device),     batch["ema"].to(device),
                batch["pros"].to(device),    batch["static"].to(device),
                batch["available"].to(device),
                act_len_ppg=batch["act_len_ppg"].to(device),
                act_len_ema=batch["act_len_ema"].to(device),
                act_len_pros=batch["act_len_pros"].to(device),
            )
            all_logits.append(logits.cpu().float())
            all_labels.append(batch["label"])
            all_pids.extend(batch["pid"].tolist())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pid_arr = np.array(all_pids)
    unique_pids = np.unique(pid_arr)

    rows = []
    for pid in unique_pids:
        mask = pid_arr == pid
        row = {"participant_id": int(pid)}
        for d, name in enumerate(tier_diseases):
            row[f"prob_{name}"] = float(probs[mask, d].mean())
            row[f"label_{name}"] = float(labels[mask][0, d])
            row[f"eligible_{name}"] = (
                int(pid in patient_filter_map[name]) if patient_filter_map else 1
            )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    protocol = "fair" if eval_fair else "unified"
    out_path = os.path.join(
        analysis_dir,
        f"patient_predictions_tier{tier}_{protocol}.csv",
    )
    df.to_csv(out_path, index=False)
    volume.commit()
    print(f"[export_checkpoint_predictions] wrote {out_path}")
    return {
        "tier": tier,
        "protocol": protocol,
        "path": out_path,
        "n_patients": int(len(df)),
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    memory=16384,
    timeout=43200,
)
def eval_checkpoint_ablation(
    tier: int = 1,
    trained_with_fair_eval: bool = True,
    eval_fair: bool = False,
    drop_modalities_csv: str = "mel",
    enabled_modalities_csv: str = "",
    run_name_suffix: str = "",
    train_profile: str = "default",
):
    """Evaluate a checkpoint after masking one or more modalities."""
    from sklearn.metrics import roc_auc_score

    assert tier in TIER_DISEASES, f"tier must be one of {sorted(TIER_DISEASES.keys())}"
    cfg = _resolve_train_config(train_profile)
    h5_path = cfg["hdf5_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drop_modalities = [m.strip() for m in drop_modalities_csv.split(",") if m.strip()]
    unknown = [m for m in drop_modalities if m not in MODALITY_KEYS]
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}")

    tier_diseases = TIER_DISEASES[tier]
    tier_enabled = _resolve_enabled_modalities(tier, enabled_modalities_csv)
    run_suffix_parts = []
    if trained_with_fair_eval:
        run_suffix_parts.append("fair_eval")
    if enabled_modalities_csv.strip():
        run_suffix_parts.append(_modality_suffix(tier_enabled))
    if run_name_suffix.strip():
        run_suffix_parts.append(run_name_suffix.strip().replace(" ", "_"))
    if train_profile.strip() and train_profile.strip() != "default":
        run_suffix_parts.append(train_profile.strip())
    tier_dir = os.path.join(cfg["ckpt_dir"], f"tier{tier}{'_' + '_'.join(run_suffix_parts) if run_suffix_parts else ''}")
    ckpt_path = os.path.join(tier_dir, "best.pt")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm_stats = {
        mod: {
            "mean": np.array(stats["mean"], dtype=np.float32),
            "std": np.array(stats["std"], dtype=np.float32),
        }
        for mod, stats in ckpt["norm_stats"].items()
    }

    with h5py.File(h5_path, "r") as hf:
        meta = hf["metadata"]
        stored_diseases = json.loads(meta.attrs["diseases"])

    disease_idx = [stored_diseases.index(d) for d in tier_diseases]
    _, _, test_loader, _ = make_dataloaders(
        h5_path, norm_stats, tier_enabled, disease_idx,
        batch_size=cfg["batch_size"], n_workers=cfg["n_workers"],
        pos_weights=None,
    )

    patient_filter_map = None
    eval_loader = test_loader
    if eval_fair:
        task_manifest = build_task_manifest(
            data_root=cfg["data_root"],
            task_names=tier_diseases,
            seed=cfg["seed"],
            val_frac_within_train=0.25,
        )
        patient_filter_map = {
            name: set(task_manifest["tasks"][name]["positive"]["test"])
            | set(task_manifest["tasks"][name]["negative"]["test"])
            for name in tier_diseases
        }
        eval_ds = _build_fair_subset(
            h5_path=h5_path,
            norm_stats=norm_stats,
            tier_enabled=tier_enabled,
            disease_idx=disease_idx,
            eligible_pid_union=set().union(*patient_filter_map.values()),
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["n_workers"],
            pin_memory=True,
            persistent_workers=(cfg["n_workers"] > 0),
        )

    model = VoiceDiseaseModel(
        n_diseases=len(tier_diseases),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        n_layers=cfg["n_transformer_layers"],
        dropout_fusion=cfg["dropout_fusion"],
        dropout_head=cfg["dropout_head"],
        head_hidden_dim=cfg.get("head_hidden_dim", 256),
        modality_dropout_prob=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    drop_idx = [MODALITY_KEYS.index(m) for m in drop_modalities]
    all_logits, all_labels, all_pids, all_tasks = [], [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            available = batch["available"].clone()
            for idx in drop_idx:
                available[:, idx] = False
            logits = model(
                batch["spec"].to(device),
                batch["mfcc"].to(device),    batch["mel"].to(device),
                batch["ppg"].to(device),     batch["ema"].to(device),
                batch["pros"].to(device),    batch["static"].to(device),
                available.to(device),
                act_len_ppg=batch["act_len_ppg"].to(device),
                act_len_ema=batch["act_len_ema"].to(device),
                act_len_pros=batch["act_len_pros"].to(device),
            )
            all_logits.append(logits.cpu().float())
            all_labels.append(batch["label"])
            all_pids.extend(batch["pid"].tolist())
            all_tasks.extend([str(t).lower() for t in batch["task_name"]])

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pid_arr = np.array(all_pids)
    unique_pids = np.unique(pid_arr)
    filter_cfg = cfg.get("task_filter_keywords", {})

    def _task_mask(disease: str) -> np.ndarray:
        keys = [k.lower() for k in filter_cfg.get(disease, [])]
        if not keys:
            return np.ones(len(all_tasks), dtype=bool)
        return np.array([any(k in t for k in keys) for t in all_tasks], dtype=bool)

    pat_probs = np.zeros((len(unique_pids), len(tier_diseases)), dtype=np.float32)
    pat_labels = np.zeros((len(unique_pids), len(tier_diseases)), dtype=np.float32)
    for d, disease in enumerate(tier_diseases):
        d_task_mask = _task_mask(disease)
        for j, pid in enumerate(unique_pids):
            pid_mask = pid_arr == pid
            use_mask = pid_mask & d_task_mask
            if not use_mask.any():
                use_mask = pid_mask
            pat_probs[j, d] = probs[use_mask, d].mean()
            pat_labels[j, d] = labels[pid_mask][0, d]

    per_disease = []
    for d, name in enumerate(tier_diseases):
        if patient_filter_map:
            keep = patient_filter_map.get(name, set())
            use = np.array([pid in keep for pid in unique_pids], dtype=bool)
        else:
            use = np.ones(len(unique_pids), dtype=bool)
        y = pat_labels[use, d]
        p = pat_probs[use, d]
        if int(y.sum()) == 0 or int((1 - y).sum()) == 0:
            per_disease.append(float("nan"))
        else:
            per_disease.append(float(roc_auc_score(y, p)))

    valid = [a for a in per_disease if not math.isnan(a)]
    macro = float(np.mean(valid)) if valid else float("nan")
    print(f"[eval_checkpoint_ablation] tier={tier} eval_fair={eval_fair} dropped={drop_modalities}")
    print(f"[eval_checkpoint_ablation] macro={macro:.4f}")
    for name, auc in zip(tier_diseases, per_disease):
        print(f"  {name:<30} {(f'{auc:.4f}' if not math.isnan(auc) else 'N/A')}")
    return {
        "tier": tier,
        "protocol": "fair" if eval_fair else "unified",
        "dropped_modalities": drop_modalities,
        "macro_auroc": macro,
        "per_disease": {n: a for n, a in zip(tier_diseases, per_disease)},
        "checkpoint_path": ckpt_path,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    memory=16384,
    timeout=43200,
)
def export_patient_counterfactuals(
    tier: int = 2,
    trained_with_fair_eval: bool = True,
    eval_fair: bool = False,
    participants_csv: str = "",
    diseases_csv: str = "",
    drop_groups_csv: str = "none;spec,mel;pros,static;ema",
    enabled_modalities_csv: str = "",
    run_name_suffix: str = "",
    train_profile: str = "default",
):
    """Export selected-patient probabilities under multiple modality-drop settings."""
    import pandas as pd

    assert tier in TIER_DISEASES, f"tier must be one of {sorted(TIER_DISEASES.keys())}"
    cfg = _resolve_train_config(train_profile)
    h5_path = cfg["hdf5_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_pids = {int(x.strip()) for x in participants_csv.split(",") if x.strip()}

    tier_diseases = TIER_DISEASES[tier]
    selected_diseases = [x.strip() for x in diseases_csv.split(",") if x.strip()] or tier_diseases
    unknown = [d for d in selected_diseases if d not in tier_diseases]
    if unknown:
        raise ValueError(f"Unknown diseases for tier{tier}: {unknown}")

    drop_groups = []
    for group in drop_groups_csv.split(";"):
        mods = [m.strip() for m in group.split(",") if m.strip()]
        if mods == ["none"] or not mods:
            mods = []
        bad = [m for m in mods if m not in MODALITY_KEYS]
        if bad:
            raise ValueError(f"Unknown modalities in group {group!r}: {bad}")
        drop_groups.append(mods)

    tier_enabled = _resolve_enabled_modalities(tier, enabled_modalities_csv)
    run_suffix_parts = []
    if trained_with_fair_eval:
        run_suffix_parts.append("fair_eval")
    if enabled_modalities_csv.strip():
        run_suffix_parts.append(_modality_suffix(tier_enabled))
    if run_name_suffix.strip():
        run_suffix_parts.append(run_name_suffix.strip().replace(" ", "_"))
    if train_profile.strip() and train_profile.strip() != "default":
        run_suffix_parts.append(train_profile.strip())
    tier_dir = os.path.join(cfg["ckpt_dir"], f"tier{tier}{'_' + '_'.join(run_suffix_parts) if run_suffix_parts else ''}")
    ckpt_path = os.path.join(tier_dir, "best.pt")
    analysis_dir = os.path.join(tier_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm_stats = {
        mod: {
            "mean": np.array(stats["mean"], dtype=np.float32),
            "std": np.array(stats["std"], dtype=np.float32),
        }
        for mod, stats in ckpt["norm_stats"].items()
    }

    with h5py.File(h5_path, "r") as hf:
        meta = hf["metadata"]
        stored_diseases = json.loads(meta.attrs["diseases"])
        labels = meta["labels"][:]
        pids = meta["pids"][:].tolist()
        all_uids = [u.decode() for u in meta["uids"][:]]
        all_task_names = [t.decode() for t in meta["task_names"][:]]

    disease_idx = [stored_diseases.index(d) for d in tier_diseases]
    patient_filter_map = None
    selected_idx = [i for i, pid in enumerate(pids) if (not selected_pids or pid in selected_pids)]
    if eval_fair:
        task_manifest = build_task_manifest(
            data_root=cfg["data_root"],
            task_names=tier_diseases,
            seed=cfg["seed"],
            val_frac_within_train=0.25,
        )
        patient_filter_map = {
            name: set(task_manifest["tasks"][name]["positive"]["test"])
            | set(task_manifest["tasks"][name]["negative"]["test"])
            for name in tier_diseases
        }
        selected_idx = [
            i for i in selected_idx
            if any(pids[i] in patient_filter_map[name] for name in selected_diseases)
        ]

    ds = VoiceDataset(
        h5_path=h5_path,
        uid_list=[all_uids[i] for i in selected_idx],
        labels=labels[selected_idx],
        norm_stats=norm_stats,
        enabled=tier_enabled,
        disease_idx=disease_idx,
        training=False,
        pids=[pids[i] for i in selected_idx],
        task_names=[all_task_names[i] for i in selected_idx],
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["n_workers"],
        pin_memory=True,
        persistent_workers=(cfg["n_workers"] > 0),
    )

    model = VoiceDiseaseModel(
        n_diseases=len(tier_diseases),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        n_layers=cfg["n_transformer_layers"],
        dropout_fusion=cfg["dropout_fusion"],
        dropout_head=cfg["dropout_head"],
        head_hidden_dim=cfg.get("head_hidden_dim", 256),
        modality_dropout_prob=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    selected_pos = {name: tier_diseases.index(name) for name in selected_diseases}
    rows = []
    with torch.no_grad():
        for batch in loader:
            pid = int(batch["pid"][0].item())
            task_name = str(batch["task_name"][0])
            for mods in drop_groups:
                available = batch["available"].clone()
                for mod in mods:
                    available[:, MODALITY_KEYS.index(mod)] = False
                logits = model(
                    batch["spec"].to(device),
                    batch["mfcc"].to(device),    batch["mel"].to(device),
                    batch["ppg"].to(device),     batch["ema"].to(device),
                    batch["pros"].to(device),    batch["static"].to(device),
                    available.to(device),
                    act_len_ppg=batch["act_len_ppg"].to(device),
                    act_len_ema=batch["act_len_ema"].to(device),
                    act_len_pros=batch["act_len_pros"].to(device),
                )
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                labels_row = batch["label"][0].cpu().numpy()
                for disease, d in selected_pos.items():
                    if patient_filter_map and pid not in patient_filter_map[disease]:
                        continue
                    rows.append({
                        "participant_id": pid,
                        "task_name": task_name,
                        "disease": disease,
                        "drop_group": "none" if not mods else "+".join(mods),
                        "prob": float(probs[d]),
                        "label": float(labels_row[d]),
                    })

    df = pd.DataFrame(rows).sort_values(["participant_id", "disease", "task_name", "drop_group"]).reset_index(drop=True)
    protocol = "fair" if eval_fair else "unified"
    tag = "all" if not selected_pids else "selected"
    out_path = os.path.join(analysis_dir, f"patient_counterfactuals_tier{tier}_{protocol}_{tag}.csv")
    df.to_csv(out_path, index=False)
    volume.commit()
    print(f"[export_patient_counterfactuals] wrote {out_path}")
    return {
        "tier": tier,
        "protocol": protocol,
        "path": out_path,
        "n_rows": int(len(df)),
    }


# =============================================================================
# SECTION 3: Local entrypoint
# =============================================================================

@app.local_entrypoint()
def main(
    tier: int = 5,
    fair_eval: bool = False,
    enabled_modalities_csv: str = "",
    run_name_suffix: str = "",
    train_profile: str = "default",
):
    """Trigger train_model for the given tier (detach-safe).

    train_model calls preprocess() internally before training, so the full
    pipeline runs on Modal's infrastructure — unaffected by local client
    disconnects or -d (detached) mode semantics.

    Examples:
        modal run -d train.py::main --tier 1
        modal run -d train.py::main --tier 2
    """
    assert tier in TIER_DISEASES, f"tier must be one of {sorted(TIER_DISEASES.keys())}, got {tier}"
    enabled = _resolve_enabled_modalities(tier, enabled_modalities_csv)
    print(f"=== Bridge2AI-Voice Training Pipeline (Tier {tier}) ===")
    print(f"  Diseases:   {TIER_DISEASES[tier]}")
    print(f"  Modalities: {[k for k, v in enabled.items() if v]}")
    print(f"  Fair eval:  {fair_eval}")
    print(f"  Profile:    {train_profile}")

    result = train_model.remote(
        tier=tier,
        fair_eval=fair_eval,
        enabled_modalities_csv=enabled_modalities_csv,
        run_name_suffix=run_name_suffix,
        train_profile=train_profile,
    )

    print("\n=== Result ===")
    print(json.dumps(result, indent=2, default=str))
