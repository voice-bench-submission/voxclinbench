# Prediction-CSV release index

Each file: `predictions/<task_id>.seed<s>.<baseline>.csv` with exactly two columns:
`subject_id,predicted_prob`. Ground-truth labels, demographics, and raw audio
paths have been filtered out (see `scripts/build_release.py` in the GitHub mirror).

## v0.4.0 coverage summary

| Category | Count | Notes |
|---|---|---|
| B2AI Tier-2 per-seed (11 diseases × 5 seeds) | 55 | Covers A1-A8 physical + A19-A21 psychiatric |
| B2AI Tier-1 / Tier-2 aggregate (fair baseline) | 2 | `b2ai.tier1_aggregate.seed0.fair.csv`, `b2ai.tier2_aggregate.seed0.fair.csv` |
| MODMA A24 zero-shot + few-shot | 2 | `modma.depression.seed0.{zero_shot,few_shot}.csv` |
| **Total shipped in v0.4.0** | **59** | Same count as v0.3.3 |

New SVD rows A10-A18 and E-DAIC PTSD row A23 baseline prediction CSVs
will follow in v0.4.1 (pilot artifacts at
`artifacts/modal_sync/svd_pilot_xlsr_20260416/` and the extend run at
`artifacts/modal_sync/svd_extend_xlsr_20260416/` are present but not
yet exported under the shipping schema).

Note: v0.3.1 shipped seven additional LODO-labelled CSVs
(`lodo.pd.test_neurovoz.seed0.{zero_shot,main,few_shot}.csv`,
`lodo.pd.test_svd.seed0.{main,unified}.csv`,
`lodo.vfp.test_svd.seed0.{main,unified}.csv`). They were removed in
v0.3.2 because cross-lingual transfer is no longer a benchmark task
category: training-corpus choice is submission-disclosed metadata, not
a benchmark-prescribed constraint. The test cohorts the LODO CSVs used
are shipped instead as external-cohort rows A9 (NeuroVoz-es PD), A10
(SVD-de VFP), and A24 (MODMA-zh depression). The zero-shot B2AI-en ->
NeuroVoz-es PD transfer (0.850 AUROC ensemble, 0.528 MARVEL baseline)
is reported in the companion paper as a descriptive finding, not as a
benchmark rank target.

| Task | Seed | Baseline | N subjects | Source artifact |
|---|---|---|---|---|
| `b2ai.tier1_aggregate` | 0 | fair | 126 | `main_v3_tier1_fair_mfcc_20260315/tier1_fair_eval/analysis/patient_predictions_tier1_unified.csv` |
| `b2ai.tier2_aggregate` | 0 | fair | 126 | `main_v3_tier2_fair_mfcc_20260315/tier2_fair_eval/analysis/patient_predictions_tier2_unified.csv` |
| `modma.depression` | 0 | few_shot | 42 | `followup_20260404/completed/external_transfer/modma_additional/modma_marvel_few_shot_depression_k5_seed0/few_shot_subject_predictions_tier2.csv` |
| `modma.depression` | 0 | zero_shot | 36 | `followup_20260404/completed/external_transfer/modma_additional/modma_marvel_zero_shot_depression/subject_predictions_tier2.csv` |

Plus 55 B2AI Tier-2 5-seed × 11-disease per-seed CSVs. In the v0.4.0
task numbering these are:
- A1-A8 physical rows (`b2ai.parkinsons, b2ai.airway_stenosis,
  b2ai.vf_paralysis, b2ai.laryngeal_dystonia, b2ai.cognitive_impair,
  b2ai.benign_lesions, b2ai.mtd, b2ai.chronic_cough`)
- A19-A21 psychiatric rows (`b2ai.depression, b2ai.ptsd, b2ai.adhd`)

Each with seeds `{0, 1, 2, 3, 4}` and suffix `.main.csv`, totalling
**59 prediction CSVs** in v0.4.0 (unchanged from v0.3.3; SVD and
E-DAIC-PTSD rows ship in v0.4.1).

Note (v0.3.3 retention): no dedicated A12 ``b2ai.psychiatric_history``
CSV is shipped. The composite A12 self-report label was formally
excluded from the scored task list in v0.3.3 (positives are a union of
A19 depression + A20 PTSD + A21 ADHD in the same cohort, so it is not
an independent biomarker); per-seed predictions for A12 were never
released as a standalone CSV, so no deletions were needed. See
CHANGELOG §0.3.3.
