# Changelog

## 0.2.0 (2026-04-13)

### Release content
- **66 per-seed prediction CSVs** under `predictions/` (11 external-
  corpus seed-0 baselines + 55 B2AI Tier-2 5-seed × 11-disease),
  each with only `subject_id,predicted_prob` columns (labels and
  demographics filtered out per upstream DUAs).
- **6 partial split manifests** under `splits/` giving test-subject
  IDs per (task, seed); train/val regeneratable via
  `voxbench.data.make_splits`.
- **Full preprocessing + model + training harness** ported into the
  package (`voxbench.data.{dataset,features,labels,splits}`,
  `voxbench.model.{branches,classifier}`, `voxbench.training`,
  `voxbench.train`, `voxbench.config`). Each preprocessing module
  docstring states explicitly that this is OUR reference recipe;
  submitters bringing their own model are not required to use it.
- **`examples/reproduce_wavlm_probe.py`**: 30-line self-contained
  recipe that reproduces the paper's WavLM-L9 macro-11 = 0.678
  baseline end-to-end from raw audio via public HuggingFace
  `microsoft/wavlm-base` weights.
- **`scripts/build_release.py` + `scripts/build_manifests.py`**:
  release-time extractors with an explicit forbidden-column deny-list
  (target, label, diagnosis, phq8, age, sex, gender, race, etc.).
- **GitHub issue + PR templates** enforcing DUA self-check and
  forbidden-column reminders.
- Datasheet (Gebru 2021 template) inlined in supplementary §13.
- **Optional `pip install voxbench[train]` extra** pulls torch,
  transformers, librosa, h5py, resampy, soundfile, modal. Core
  `voxbench eval` / `voxbench compare` still ships on numpy + scipy
  + scikit-learn only.

### Harness
- Replaced bootstrap-based `delong_p` placeholder with a proper
  DeLong U-statistic kernel (Sun & Xu 2014 fast O(N log N) kernel).
- `delong_p` now returns `(auc_diff, p_value)` and validates inputs
  (NaN, single-class labels, shape mismatch).
- `paired_bootstrap_p` as non-parametric sanity check.
- `voxbench compare --test delong` (default) / `--test paired-bootstrap`.
- `voxbench eval` and `voxbench compare` accept CSV submissions with
  two-column `subject_id,predicted_prob` schema; pass `--labels`
  pointing to a CSV of subject-level labels obtained from the
  credentialed upstream corpus. JSON submissions with full
  `subject_probs` dict also supported.
- `bootstrap_ci` gracefully returns `(nan, nan)` for degenerate
  inputs (N<2 or single-class) instead of raising.
- `voxbench.data.make_splits` deterministic participant-wise
  stratified split function with 6 regression tests.
- Added `voxbench/__main__.py` so `python -m voxbench --help` works.

### Documentation / metadata
- Corrected upstream license metadata after a fresh audit:
  - SVD is publicly downloadable via Zenodo mirror (CC BY 4.0,
    records 16874898 + 7024894); not a Saarland access form.
  - DAIC-WOZ and E-DAIC: EULA governs use; files at
    dcapswoz.ict.usc.edu are HTTP-reachable without login.
  - NeuroVoz: CC BY-NC-ND 4.0 (was erroneously CC BY 4.0); Zenodo
    files are access-restricted.
- README documents two workflows explicitly:
  (A) submit your own model's predictions (core voxbench),
  (B) reproduce / retrain VoxClinBench-Base (voxbench[train]).
- Croissant 1.0 metadata file mirrored on HuggingFace Datasets
  (`croissant.json`) per NeurIPS 2026 E&D requirements.

### Verified
- End-to-end smoke test on a fresh venv: git clone → pip install →
  voxbench eval on released CSV → AUROC/AUPRC/95%CI JSON. Workflow A
  runs without torch; workflow B reproduces WavLM-L9 probe on toy
  synthetic audio (real raw wav → HF WavLM → LogReg → CSV →
  voxbench eval).
- `pytest` 13/13 pass on every commit.

## 0.1.0 (initial)

- Task registry (22 + 1 scoping tasks), DeLong + bootstrap CI
  primitives, CLI scaffolding.
