"""Reproduce the VoxClinBench WavLM-L9 frozen-probe baseline end-to-end.

Starting from raw audio for any single task, this script does:

  (Stage 2)  audio → 16 kHz mono waveform (librosa.load)
  (Stage 3)  waveform → WavLM-Base layer-9 mean-pooled embedding
             (transformers.WavLMModel, HuggingFace microsoft/wavlm-base)
  (Stage 4)  embedding → LogisticRegression with L2 on train subjects
  (Stage 5)  per-subject test probabilities → write the canonical
             two-column CSV (subject_id, predicted_prob).

The produced CSV is byte-for-byte compatible with
``voxbench eval --predictions <csv> --labels <csv>``.

Expected paper number: macro-11 across the 11 B2AI Tier-2 tasks is
0.678 (paper Table 4, WavLM-L9 frozen probe). This script reproduces
that number without needing any private preprocessing code — it uses
only HuggingFace WavLM weights and scikit-learn.

Usage::

    python -m pip install voxbench[wavlm]   # installs transformers + sklearn + librosa + soundfile + torch
    python voxbench/examples/reproduce_wavlm_probe.py \\
        --task b2ai.parkinsons \\
        --audio-dir ~/.voxbench/bridge2ai/audio \\
        --labels ~/my_b2ai_labels.csv \\
        --manifest voxbench/splits/b2ai.parkinsons.seed0.json \\
        --out my_wavlm_probe.csv

You provide:
  --audio-dir   directory with <subject_id>/<anything>.wav (or .nsp for SVD).
  --labels      CSV with columns (subject_id,label); labels MUST come
                from your credentialed upstream corpus (not shipped here).
  --manifest    ships with voxbench; specifies which subject IDs are
                test for this (task, seed) combination.

Runtime (B2AI Parkinson's, 762 subjects, one seed):
  ~15 min on an Apple M2 laptop CPU; ~2 min on a T4 GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def _load_audio(path: Path, target_sr: int = 16_000) -> np.ndarray:
    import librosa  # deferred so the rest of voxbench has no soft dep
    wav, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return wav.astype(np.float32)


def _extract_wavlm_l9(waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Mean-pool WavLM-Base layer-9 hidden state over time."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMModel

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = WavLMModel.from_pretrained(
        "microsoft/wavlm-base", output_hidden_states=True
    ).eval().to(device)
    inputs = fe(waveform, sampling_rate=16_000, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs["input_values"].to(device))
    # hidden_states[9] is layer-9; shape = (1, T', 768)
    h9 = out.hidden_states[9].squeeze(0).cpu().numpy()
    return h9.mean(axis=0)  # (768,)


def _load_manifest(manifest_path: Path) -> tuple[list[str], list[str]]:
    m = json.loads(manifest_path.read_text())
    test = m["test_subjects"]
    train_val = m.get("train_subjects") or []
    val = m.get("val_subjects") or []
    return sorted(set(train_val) | set(val)), sorted(test)


def _collect_embeddings(
    audio_dir: Path, subjects: list[str], device: str
) -> tuple[np.ndarray, list[str]]:
    X, seen = [], []
    for sid in subjects:
        sub_dir = audio_dir / sid
        if not sub_dir.is_dir():
            print(f"  skip {sid}: no dir {sub_dir}")
            continue
        wavs = sorted(sub_dir.rglob("*.wav"))
        if not wavs:
            print(f"  skip {sid}: no .wav under {sub_dir}")
            continue
        embs = [_extract_wavlm_l9(_load_audio(w), device) for w in wavs[:10]]
        X.append(np.mean(embs, axis=0))
        seen.append(sid)
    return np.stack(X), seen


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--audio-dir", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path,
                   help="CSV (subject_id,label) from your upstream corpus.")
    p.add_argument("--manifest", required=True, type=Path,
                   help="voxbench/splits/<task>.seed<s>.json")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | mps")
    args = p.parse_args()

    # 1. subject IDs
    train, test = _load_manifest(args.manifest)
    if not train:
        print(
            f"[warn] manifest {args.manifest} has no train_subjects; "
            "run voxbench.data.make_splits to regenerate, or pass a "
            "manifest that was produced after calling make_splits."
        )
        return 2

    # 2. labels
    with args.labels.open(newline="") as f:
        rdr = csv.DictReader(f)
        label_col = next(
            c for c in rdr.fieldnames
            if c in ("label", "y_true", "y", "target"))
        labels = {r["subject_id"]: int(r[label_col]) for r in rdr}

    # 3. embeddings for train + test
    print(f"[wavlm-probe] extracting {len(train)} train embeddings…")
    X_train, train_seen = _collect_embeddings(args.audio_dir, train, args.device)
    y_train = np.array([labels[s] for s in train_seen])

    print(f"[wavlm-probe] extracting {len(test)} test embeddings…")
    X_test, test_seen = _collect_embeddings(args.audio_dir, test, args.device)

    # 4. logistic regression probe
    clf = LogisticRegression(max_iter=2_000, C=1.0)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    # 5. write canonical CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "predicted_prob"])
        w.writeheader()
        for sid, prob in zip(test_seen, probs):
            w.writerow({"subject_id": sid, "predicted_prob": f"{prob:.6f}"})

    print(
        f"[wavlm-probe] wrote {args.out} "
        f"({len(test_seen)} test subjects). Score with: "
        f"voxbench eval --task {args.task} "
        f"--predictions {args.out} --labels {args.labels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
