#!/usr/bin/env python3
"""Phase 2 extractor: assemble canonical prediction CSVs for VoxClinBench release.

Input:  $VOXBENCH_ARTIFACTS (default: ./artifacts/modal_sync/)
Output: $VOXBENCH_RELEASE/predictions/<task_id>.seed<s>.<baseline>.csv
         (default output root: ./release/)
         with columns: subject_id,predicted_prob
Also writes $VOXBENCH_RELEASE/predictions/INDEX.md summarising coverage.

Safety:
- DROPS any column named target / label / diagnosis / phq8 / age / sex /
  gender / race / ethnicity / subset / site_id from every CSV before
  output (NeurIPS DUA compliance).
- Keeps only {subject_id, probability} columns.
- Never pushes raw audio paths or demographic metadata.
"""

from __future__ import annotations
import csv
import os
import re
import sys
from pathlib import Path

ARTIFACTS = Path(os.environ.get("VOXBENCH_ARTIFACTS", "./artifacts/modal_sync")).resolve()
OUT_ROOT = Path(os.environ.get("VOXBENCH_RELEASE", "./release")).resolve()
OUT_PRED = OUT_ROOT / "predictions"
OUT_PRED.mkdir(parents=True, exist_ok=True)

# columns we ALWAYS drop
FORBIDDEN = {
    "target", "label", "labels", "diagnosis", "y_true", "y",
    "phq8", "phq_8", "phq8_score", "phq_score", "score",
    "age", "sex", "gender", "race", "ethnicity", "demographics",
    "subset", "site", "site_id", "cohort", "split",
}

# columns we KEEP as probability
PROB_COLS = re.compile(r"^(prob|pred|predicted|probability|score_main|p_|logit)", re.I)

# Map artifact directory name patterns to canonical task_id
DIR_TO_TASK = [
    (re.compile(r"external_transfer_svd_vf_paralysis"), "lodo.vfp.test_svd"),
    (re.compile(r"external_transfer_svd_parkinsons"),   "lodo.pd.test_svd"),   # informal, not in 22
    (re.compile(r"neurovoz_marvel|neurovoz_transfer|neurovoz_results"), "lodo.pd.test_neurovoz"),
    (re.compile(r"modma_reverse"),                      "lodo.dep.test_modma_reverse"),
    (re.compile(r"modma_marvel"),                       "modma.depression"),
    (re.compile(r"main_v3_tier1"),                      "b2ai.tier1_aggregate"),
    (re.compile(r"main_v3_tier2"),                      "b2ai.tier2_aggregate"),
    (re.compile(r"3mod_external_eval"),                 "depression.lodo.3mod"),
]

def detect_task(path: Path) -> str | None:
    s = str(path)
    for pat, task in DIR_TO_TASK:
        if pat.search(s):
            return task
    return None

def detect_seed(path: Path) -> int | None:
    m = re.search(r"seed[_=]?(\d+)", str(path))
    return int(m.group(1)) if m else None

def detect_baseline(path: Path) -> str:
    s = str(path).lower()
    if "zero_shot" in s: return "zero_shot"
    if "few_shot" in s: return "few_shot"
    if "adapted" in s: return "adapted"
    if "fair" in s: return "fair"
    if "unified" in s: return "unified"
    return "main"

def filter_csv(src: Path) -> tuple[list[dict], list[str]]:
    """Return (rows, kept_cols) after filtering PHI/label columns."""
    with open(src, newline='') as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            return [], []
        kept = [c for c in rdr.fieldnames
                if c.lower() not in FORBIDDEN
                and (c == "subject_id" or c == "participant_id" or PROB_COLS.match(c))]
        # must have subject_id and at least one prob column
        if not any(c == "subject_id" or c == "participant_id" for c in kept):
            return [], []
        if not any(PROB_COLS.match(c) for c in kept):
            return [], []
        rows = [{k: r.get(k, "") for k in kept} for r in rdr]
    return rows, kept

def canonicalise(rows: list[dict], cols: list[str]) -> list[dict]:
    """Emit uniform schema: subject_id, predicted_prob (choose first prob col)."""
    sid = "subject_id" if "subject_id" in cols else "participant_id"
    prob_col = next(c for c in cols if PROB_COLS.match(c))
    return [{"subject_id": r[sid], "predicted_prob": r[prob_col]}
            for r in rows if r.get(sid)]

def main():
    extracted = []
    for csv_path in ARTIFACTS.rglob("subject_predictions*.csv"):
        if csv_path.stat().st_size < 200:  # skip empty
            continue
        task = detect_task(csv_path)
        if task is None:
            continue
        seed = detect_seed(csv_path) or 0
        bl = detect_baseline(csv_path)
        rows, cols = filter_csv(csv_path)
        if not rows:
            continue
        canon = canonicalise(rows, cols)
        out_name = f"{task}.seed{seed}.{bl}.csv"
        out_path = OUT_PRED / out_name
        with open(out_path, "w", newline='') as f:
            w = csv.DictWriter(f, fieldnames=["subject_id", "predicted_prob"])
            w.writeheader()
            w.writerows(canon)
        extracted.append({
            "task": task, "seed": seed, "baseline": bl,
            "n_subjects": len(canon),
            "src": str(csv_path.relative_to(ARTIFACTS)),
            "out": out_name,
        })

    # Also grab few_shot and zero_shot that weren't named subject_predictions*
    for csv_path in ARTIFACTS.rglob("*subject_predictions*.csv"):
        if csv_path in [Path(e["out"]) for e in extracted]:
            continue
        task = detect_task(csv_path)
        if task is None: continue
        seed = detect_seed(csv_path) or 0
        bl = detect_baseline(csv_path)
        rows, cols = filter_csv(csv_path)
        if not rows: continue
        canon = canonicalise(rows, cols)
        out_name = f"{task}.seed{seed}.{bl}.csv"
        if (OUT_PRED / out_name).exists(): continue
        with open(OUT_PRED / out_name, "w", newline='') as f:
            w = csv.DictWriter(f, fieldnames=["subject_id", "predicted_prob"])
            w.writeheader()
            w.writerows(canon)
        extracted.append({
            "task": task, "seed": seed, "baseline": bl,
            "n_subjects": len(canon),
            "src": str(csv_path.relative_to(ARTIFACTS)),
            "out": out_name,
        })

    # Also scan patient_predictions_*.csv
    for csv_path in ARTIFACTS.rglob("patient_predictions*.csv"):
        task = detect_task(csv_path)
        if task is None: continue
        seed = detect_seed(csv_path) or 0
        bl = detect_baseline(csv_path)
        rows, cols = filter_csv(csv_path)
        if not rows: continue
        canon = canonicalise(rows, cols)
        out_name = f"{task}.seed{seed}.{bl}.csv"
        if (OUT_PRED / out_name).exists(): continue
        with open(OUT_PRED / out_name, "w", newline='') as f:
            w = csv.DictWriter(f, fieldnames=["subject_id", "predicted_prob"])
            w.writeheader()
            w.writerows(canon)
        extracted.append({
            "task": task, "seed": seed, "baseline": bl,
            "n_subjects": len(canon),
            "src": str(csv_path.relative_to(ARTIFACTS)),
            "out": out_name,
        })

    # Write index
    lines = ["# Prediction-CSV release index",
             "",
             "Each file: `predictions/<task_id>.seed<s>.<baseline>.csv` with exactly two columns:",
             "`subject_id,predicted_prob`. Ground-truth labels, demographics, and raw audio",
             "paths have been filtered out (see `scripts/build_release.py` in the GitHub mirror).",
             "",
             "| Task | Seed | Baseline | N subjects | Source artifact |",
             "|---|---|---|---|---|"]
    for e in sorted(extracted, key=lambda x: (x["task"], x["seed"], x["baseline"])):
        lines.append(f"| `{e['task']}` | {e['seed']} | {e['baseline']} | {e['n_subjects']} | `{e['src']}` |")
    with open(OUT_PRED / "INDEX.md", "w") as f:
        f.write("\n".join(lines))

    print(f"Extracted {len(extracted)} prediction CSVs to {OUT_PRED}")
    for e in extracted[:5]:
        print(f"  {e['out']}: n={e['n_subjects']} from {e['src']}")

if __name__ == "__main__":
    main()
