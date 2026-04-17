# Changelog

## [0.4.1] — 2026-04-16

### Changed
- Task ID `A26 modma.depression` renumbered to `A24` (cross-lingual
  reference row, WavLM-L9 zero-shot from E-DAIC).
- A24 MODMA value: legacy single-seed 0.799 superseded by 5-seed
  WavLM-L9 zero-shot mean 0.680 ± 0.092 (ddof=0). Legacy used mixed
  head / bootstrap CI; the new value is methodologically consistent
  with A22.
- A9 `neurovoz.parkinsons`: 3-seed 0.955 ± 0.041 → 5-seed 0.936 ± 0.039
  (seeds [42, 1, 2, 3, 4]).
- A10 `svd.vf_paralysis`: 3-seed 0.929 ± 0.021 → 5-seed 0.936 ± 0.019
  (seeds [42, 1, 2, 3, 4]).
- `macro_physical_18` recomputed with updated A9/A10 values; see
  `artifacts/eda/dual_family_macro.json` (mean 0.853, SD 0.107 at 3dp;
  raw values 0.8528 / 0.1067 with ddof=1).
- §Methods backbone policy: removed "consistency-preferred"
  pre-declaration; the reported baseline is now "reasonable-effort
  per-task with whichever SSL backbone we evaluated with sufficient
  seeds." Per-task evaluated-backbone matrix added as supplementary
  §\ref{ssec:backbones-matrix}.
- Main + supplementary text: removed interpretive framing (e.g.
  "robustness", "surprisingly strong", "demonstrates"). Benchmark
  reports numbers, not interpretations.

### Removed
- Task IDs `A24` (edaic.phq8_regression) and `A25` (edaic.ptsd_severity)
  retired from the benchmark registry. Their 5-seed Pearson r results
  are preserved in the supplementary "Exploratory Regression Analyses"
  section without task IDs.

### Total registry size
- **24 entries** (23 scored binary tasks + 1 cross-lingual reference
  row A24 MODMA, outside the Holm family). Down from 26 registrations
  in v0.4.0 (the two regression scoping entries were retired).

### Retention ledger
- 2026-04-16: A26 → A24 (MODMA cross-lingual reference renumbering);
  legacy 0.799 mean superseded by 0.680 (WavLM-L9 5-seed zero-shot).
- 2026-04-16: old A24/A25 regression tasks demoted to supplementary
  exploration.

## 0.4.0 (2026-04-16)

Major task-roster expansion: **26 registrations** (**23 scored** + 2
regression scoping + 1 zero-shot cross-lingual validation target), up
from 16 registrations / 15 scored in v0.3.3. Corpus count stays at 5;
language coverage stays at 4 (en, es, de, zh). Clean A1-A24 numbering
replaces the former A12 / A15 / A16a / A16b / A17 suffix notation.

### Added
- **8 new SVD German-language pathology tasks (A11-A18)**, each scored
  as an external-cohort row (AUROC, n_pos from the Saarland SVD
  release, shared n_neg=300 healthy pool):
  - A11 `svd.hyperfunctional_dysphonia` (Hyperfunktionelle Dysphonie),
    n_pos=199
  - A12 `svd.laryngitis` (Laryngitis), n_pos=128
  - A13 `svd.functional_dysphonia` (Funktionelle Dysphonie), n_pos=108
  - A14 `svd.psychogenic_dysphonia` (Psychogene Dysphonie), n_pos=80
  - A15 `svd.contact_pachydermia` (Kontaktpachydermie), n_pos=63
  - A16 `svd.reinke_edema` (Reinke Ödem), n_pos=54
  - A17 `svd.dysodia` (Dysodie), n_pos=54
  - A18 `svd.vf_polyp` (Stimmlippenpolyp), n_pos=40
- **A23 `edaic.ptsd`**: E-DAIC English PTSD binary row derived from the
  PCL-C cutoff column (`PTSD_label` in E-DAIC `detailed_lables.csv`).
  Scored AUROC leaderboard row in Family B.
- **A25 `edaic.ptsd_severity`**: E-DAIC PCL-C severity regression
  scoping entry (Pearson r / CCC; label column `PTSD_severity`,
  PCL-C total score in range 17-85). Kept in the registry for
  reproducibility; not a scored leaderboard row.
- All new SVD tasks evaluated with XLSR-53 layer 18 frozen + LogReg
  baseline (5 seeds, 80/20 subject-stratified), pilot artifacts at
  `artifacts/modal_sync/svd_pilot_xlsr_20260416/` and extend-run
  artifacts at `artifacts/modal_sync/svd_extend_xlsr_20260416/`.
- **Croissant distribution list**: one `cr:FileObject` per SVD
  pathology ZIP pointing at the primary Zenodo archive
  (record 16874898, CC-BY-4.0). Makes the upstream source of each
  SVD row individually addressable from the Croissant graph.
- **`list_scorable_tasks()` also filters out the (then-A26, now-A24)
  MODMA row** (previously only filtered the A24 PHQ-8 regression
  scoping entry). The MODMA row is a zero-shot cross-lingual
  reference, not a Holm-family scored ranking; adding A23 and A25 to
  the registry made the previous filter lossy, so we made the rule
  explicit. The scoring set remains Family-A-scored + Family-B-scored
  only.

### Changed
- **Task count**: registrations 16 -> 26 (scored 15 -> 23; scoping
  1 -> 2; zero-shot target 0 -> 1, was previously counted inside the
  15 scored rows).
- **B2AI share of scored tasks**: 11/23 ≈ 48% (was 11/15 ≈ 73% in
  v0.3.3). Corpus diversity goal met — no single corpus now dominates
  the scored roster.
- **Task numbering resequenced** to clean A1-A24. The former A12
  (psychiatric_history), A15 (daicwoz.depression), A16a/A16b (split
  regression notation), and A17 LODO labels are retired; they only
  survive in this CHANGELOG's retraction ledger and in git history.
- **NeuroVoz split manifest**: switched from a redistributed CSV to a
  user-generates-locally script at `voxbench/splits/neurovoz_splitter.py`,
  to respect the CC-BY-NC-ND-4.0 license ND clause (a redistributed
  subject-ID split manifest is a derivative work; a script that
  regenerates the same split deterministically from the user's own
  fetched copy of NeuroVoz is not).
- **CHANGELOG `predictions/INDEX.md` claim**: per-seed CSV coverage
  updated from "A1-A21 disease rows + A24 MODMA" to "A1-A8 + A19-A21
  B2AI Tier-2 rows (55 CSVs, 5 seeds × 11 diseases) + 2 B2AI aggregate
  rows + A24 MODMA (1 few-shot + 1 zero-shot)". New A10-A18 SVD rows
  ship seed-0-only baseline predictions initially; 5-seed prediction
  CSVs will follow in v0.4.1 once the extend run completes.
- Datasheet, Croissant `rai:dataLimitations`, issue-template submission
  checkbox list, and `SUBMISSION.md` / `README.md` updated for the new
  counts.

### Removed
- `svd.leukoplakia` (n_pos=34 < n_pos≥40 scoring threshold)
- `svd.presbyphonia` (n_pos=39 < threshold + age-HC confound: the
  SVD presbyphonia cohort is elderly-only, so AUROC against the shared
  healthy pool would primarily score age rather than dysphonia)
- Redistributed NeuroVoz subject-ID manifest CSV(s); replaced by the
  deterministic splitter script (see above).

### Fixed
- (nothing; v0.4.0 is an additive roster expansion + NeuroVoz license
  compliance fix, not a bug-fix release)

### Retraction ledger (v0.4.0 snapshot)
Tasks that were once in a VoxClinBench release and are NOT in v0.4.0:
- **old A12** `b2ai.psychiatric_history` — excluded v0.3.3 (composite
  self-report, positives are a union of A19 depression + A20 PTSD +
  A21 ADHD in the same cohort; not an independent biomarker).
- **old A15** `daicwoz.depression` — retracted v0.3.1 (DAIC-WOZ
  subjects are a 189/189 strict subset of E-DAIC; not independent).
- **LODO rows** `lodo.pd.test_neurovoz` (B1),
  `lodo.vfp.test_svd` (B6), and the depression LODO family
  (B2 `lodo.dep.test_b2ai`, B3 `lodo.dep.test_daic`,
  B4 `lodo.dep.test_edaic`, B5 `lodo.dep.test_modma`) — removed
  v0.3.2 (training-corpus choice is submission-disclosed, not
  benchmark-prescribed, so LODO training protocols do not define
  benchmark rows; test cohorts retained as A9, A10, A22, A24).
- **SVD leukoplakia** and **SVD presbyphonia** — scoped out v0.4.0
  (see Removed above).

## 0.3.3 (2026-04-16)

Excluded B2AI's composite A12 `psychiatric_history` label from the
scored task list; scored-task count corrected 16 -> 15 (11 within-
Bridge2AI: A1-A8 physical + A19-A21 psychiatric; 4 external-cohort:
A9 NeuroVoz-es, A10 SVD-de, A22 E-DAIC-en depression, A24 MODMA-zh).
Family B psychiatric macro is now ``mean{A19 depression, A20 PTSD,
A21 ADHD}`` (n=3, was n=4); with VoxClinBench-Base seed-mean values
AUROC_A9=0.700, AUROC_A10=0.707, AUROC_A11=0.619, macro_psychiatric =
0.675 (was 0.684 with A12 folded in). Physical-vs-psychiatric macro
gap widens to ~16 points. Benchmark paper v3 PDFs, Croissant metadata,
submission issue template, and task-registry docstring updated.

### Why A12 was excluded as a scored task

B2AI's `psychiatric_history` column is a composite self-report label
whose positive set is the union of the depression / PTSD / ADHD /
anxiety / bipolar self-reports in the same cohort. Its positives are
therefore a superset of A19 depression + A20 PTSD + A21 ADHD by
construction -- not an independent biomarker. The VoxClinBench-Base
AUROC on A12 (0.708 in pre-v0.3.3 drafts) is arithmetically consistent
with being a noisy union of the A19-A21 signals and adds no information
beyond them. The paper's earlier text contradicted itself (Task
Inclusion §Method said A12 was excluded; the abstract counted it in
the psychiatric macro); v0.3.3 resolves the contradiction by applying
the Task Inclusion rule everywhere.

### Registry / release-content delta (2026-04-16)
- **No new CSVs deleted.** A12 was never shipped as a standalone
  prediction CSV -- per-seed prediction CSVs only exist for the scored
  within-B2AI disease rows (A1-A8 physical + A19-A21 psychiatric) and
  A24 MODMA. Prediction CSV count stays at **59**.
- **No task-registry hard-deletion.** A12 was never present as a
  `Task(...)` entry in `voxbench/tasks.py`; its exclusion was already
  implicit in the registry but is now stated explicitly in the module
  docstring.
- Added `voxbench.tasks.list_scorable_tasks()` returning the 15 scored
  tasks (excludes the A24 PHQ-8 regression scoping entry, which
  remains in `TASKS` for reproducibility but is not a leaderboard row).
- `croissant.json` `version` bumped 0.3.2 -> 0.3.3; `description`,
  tasks recordSet description, and `rai:dataLimitations` updated to
  say 15 scored tasks (11 within-B2AI + 4 external-cohort) and to
  call out the A12 exclusion.
- `README.md`, `scripts/build_release.py`, `predictions/INDEX.md`,
  `splits/INDEX.md`, `.github/ISSUE_TEMPLATE/submission.md`,
  `SUBMISSION.md` updated for the new counts.

## 0.3.2 (2026-04-16)

Removed cross-lingual leave-one-dataset-out (LODO) as a benchmark task
category; task count corrected 18 -> 16 live (12 within-Bridge2AI +
4 external-cohort); LODO prediction and split files
(`lodo.pd.*`, `lodo.vfp.*`) dropped from the release; benchmark paper
v3 PDFs and Croissant metadata reflect the new framing.

### Why LODO was removed as a task

A benchmark task specifies (data, labels, metric). Which corpora a
submission trains on -- English-only B2AI, multilingual pool, target-
corpus fine-tune -- is submission-disclosed metadata, not a
benchmark-prescribed constraint. The v0.3.1 "Cross-lingual LODO"
rows (B1 PD EN->ES, B6 VFP EN->DE) conflated these two concepts:
their test halves are identical to the external-cohort rows A9
(NeuroVoz-es PD) and A10 (SVD-de VFP) that are already first-class
benchmark tasks, and their training halves prescribed a specific
submission choice. Submitters may still report zero-shot cross-lingual
AUROC (e.g. 0.850 AUROC B2AI-en -> NeuroVoz-es PD, Finding 1 in the
paper) as a descriptive finding; it is not a benchmark rank target.

### Task registry cleanup (2026-04-16)
- **Hard-deleted the remaining LODO task entries** from
  `voxbench/tasks.py`:
  - `lodo.pd.test_neurovoz` (B1): test cohort already shipped as A9.
  - `lodo.vfp.test_svd` (B6): test cohort already shipped as A10.
- Removed the `LODO_TASKS` module-level grouping and the
  `Cross-lingual LODO` section of the task list.
- `TaskKind` literal narrowed from `{"within", "lodo"}` to
  `{"within", "external"}`; NeuroVoz, SVD, E-DAIC, MODMA rows are now
  labelled `external`.
- Dropped `Task.lodo_train_corpora` field; training-corpus declaration
  moves to submission metadata.
- Task count: 18 -> 16 live (12 within-Bridge2AI + 4 external-cohort).

### Release-content delta (2026-04-16)
- **Deleted 7 LODO prediction CSVs** from `predictions/`:
  `lodo.pd.test_neurovoz.seed0.{zero_shot,main,few_shot}.csv`,
  `lodo.pd.test_svd.seed0.{main,unified}.csv`,
  `lodo.vfp.test_svd.seed0.{main,unified}.csv`. Prediction CSV count:
  66 -> 59.
- **Deleted 3 LODO split manifests** from `splits/`:
  `lodo.pd.test_neurovoz.seed0.json`,
  `lodo.pd.test_svd.seed0.json`,
  `lodo.vfp.test_svd.seed0.json`. Split manifest count: 6 -> 3.
- `predictions/INDEX.md` and `splits/INDEX.md` regenerated; `croissant.json`
  description and `rai:dataLimitations` updated to drop 18-task +
  LODO-B1-B6 references; issue-template submission checkbox for
  "Cross-lingual LODO (B1--B6)" removed; `scripts/build_release.py`
  task-map patterns updated.

## 0.3.1 (2026-04-14)

CLGP dual-family reframe; DAIC-WOZ + depression LODO B2-B5 retracted
(189/189 E-DAIC overlap); task count corrected 17 -> 18 live
(16 within-corpus + 2 LODO); benchmark paper v3 PDFs shipped under
`paper/main_voxbench.pdf` and `paper/supplementary_voxbench.pdf`.

### Task registry cleanup (2026-04-14)
- **Hard-deleted 5 retracted task entries** from `voxbench/tasks.py`:
  - `daicwoz.depression` (A15): retracted because DAIC-WOZ subjects are a
    strict subset of E-DAIC (189/189 overlap), making it non-independent.
  - `lodo.dep.test_b2ai` (B2), `lodo.dep.test_daic` (B3),
    `lodo.dep.test_edaic` (B4), `lodo.dep.test_modma` (B5): depression
    LODO family contaminated by the DAIC-WOZ subset of E-DAIC and by
    target corpora appearing in the train set. Superseded by A19, A22,
    A24 within-corpus results.
- Task count: 23 -> 18 live (16 within-corpus + 2 LODO: B1 PD EN->ES on
  NeuroVoz, B6 VFP EN->DE on SVD). _Superseded by v0.3.2: LODO rows
  B1/B6 subsequently removed as benchmark tasks, see entry above._
- Retraction ledger preserved in paper/voxbench_paper/snippets/task_card.tex
  and supp \S\ref{sec:lodo-audit}; audit trail available via git history.

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
