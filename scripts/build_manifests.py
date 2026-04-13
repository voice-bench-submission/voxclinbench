#!/usr/bin/env python3
"""Generate partial split manifests from released prediction CSVs.

For each released predictions/<task>.seed<s>.<baseline>.csv, extract the
unique subject IDs (these are the TEST subjects for that task/seed) and
write a manifest JSON with:
    {"task_id": ..., "seed": ..., "test_subjects": [...]}

train/val subjects are deterministically regeneratable from the upstream
labels via `voxbench.data.make_splits(task_id, seed)` once the user has
fetched the credentialed source corpus — this keeps the released
manifest honest: we publish exactly what we can prove (test set) and
refer users to a reproducible regeneration path for the rest.
"""
from __future__ import annotations
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = Path(os.environ.get("VOXBENCH_PREDICTIONS", REPO_ROOT / "predictions")).resolve()
OUT_DIR = Path(os.environ.get("VOXBENCH_SPLITS", REPO_ROOT / "splits")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Group predictions by (task, seed) — one manifest per (task, seed),
# merging test subjects across baselines (they should all have the same
# test set).
groups: dict[tuple[str, int], set[str]] = defaultdict(set)
sources: dict[tuple[str, int], list[str]] = defaultdict(list)

for csv_path in sorted(PRED_DIR.glob("*.csv")):
    m = re.match(r"^(.+)\.seed(\d+)\.(.+)\.csv$", csv_path.name)
    if not m:
        continue
    task, seed, baseline = m.group(1), int(m.group(2)), m.group(3)
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        ids = {row["subject_id"] for row in rdr if row.get("subject_id")}
    if not ids:
        continue
    groups[(task, seed)].update(ids)
    sources[(task, seed)].append(f"{baseline} ({len(ids)} subjects)")

for (task, seed), ids in sorted(groups.items()):
    manifest = {
        "task_id": task,
        "seed": seed,
        "test_subjects": sorted(ids),
        "train_subjects": None,
        "val_subjects": None,
        "provenance": {
            "test_subjects_derived_from": "released prediction CSVs",
            "train_val_derivation": "run `voxbench.data.make_splits(task_id, seed)` after `voxbench fetch <corpus>`; split is deterministic given the upstream subject list and seed.",
            "baselines_contributing_test_ids": sources[(task, seed)],
        },
    }
    out_path = OUT_DIR / f"{task}.seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

# Index
with open(OUT_DIR / "INDEX.md", "w") as f:
    f.write("# Split manifests — partial (test-set only)\n\n")
    f.write(f"{len(groups)} manifests, one per (task, seed) extracted from the released prediction CSVs.\n\n")
    f.write("| Task | Seed | N test | Baselines contributing |\n|---|---|---|---|\n")
    for (task, seed), ids in sorted(groups.items()):
        f.write(f"| `{task}` | {seed} | {len(ids)} | {', '.join(sources[(task, seed)])} |\n")
    f.write("\n## How to reproduce train/val splits\n\n")
    f.write("Once the upstream corpus is fetched (`voxbench fetch <corpus>`), call:\n\n")
    f.write("```python\nfrom voxbench.data import make_splits\nsplit = make_splits(task_id=\"b2ai.parkinsons\", seed=0)\n# returns {\"train\": [...], \"val\": [...], \"test\": [...]}\n```\n\n")
    f.write("The test-subject list returned should match this manifest's `test_subjects` field byte-for-byte.\n")

print(f"Wrote {len(groups)} manifests to {OUT_DIR}")
