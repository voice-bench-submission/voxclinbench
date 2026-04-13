# Prediction-CSV release index

Each file: `predictions/<task_id>.seed<s>.<baseline>.csv` with exactly two columns:
`subject_id,predicted_prob`. Ground-truth labels, demographics, and raw audio
paths have been filtered out (see `scripts/build_release.py` in the GitHub mirror).

| Task | Seed | Baseline | N subjects | Source artifact |
|---|---|---|---|---|
| `b2ai.tier1_aggregate` | 0 | fair | 126 | `main_v3_tier1_fair_mfcc_20260315/tier1_fair_eval/analysis/patient_predictions_tier1_unified.csv` |
| `b2ai.tier2_aggregate` | 0 | fair | 126 | `main_v3_tier2_fair_mfcc_20260315/tier2_fair_eval/analysis/patient_predictions_tier2_unified.csv` |
| `lodo.pd.test_neurovoz` | 0 | few_shot | 98 | `followup_20260404/completed/external_transfer/neurovoz_results/neurovoz_marvel_few_shot_parkinsons_k5_seed0/few_shot_subject_predictions_tier1.csv` |
| `lodo.pd.test_neurovoz` | 0 | main | 108 | `followup_20260404/completed/external_transfer/neurovoz_results/neurovoz_transfer_parkinsons/subject_predictions_tier1.csv` |
| `lodo.pd.test_neurovoz` | 0 | zero_shot | 98 | `followup_20260404/completed/external_transfer/neurovoz_results/neurovoz_marvel_few_shot_parkinsons_k5_seed0/zero_shot_subject_predictions_tier1.csv` |
| `lodo.pd.test_svd` | 0 | main | 32 | `external_transfer_svd_parkinsons_20260315/svd_transfer_parkinsons/subject_predictions_tier1.csv` |
| `lodo.pd.test_svd` | 0 | unified | 32 | `external_transfer_svd_parkinsons_unified_main_20260315/svd_transfer_parkinsons/subject_predictions_tier1.csv` |
| `lodo.vfp.test_svd` | 0 | main | 53 | `external_transfer_svd_vf_paralysis_20260315/svd_transfer_vf_paralysis/subject_predictions_tier1.csv` |
| `lodo.vfp.test_svd` | 0 | unified | 53 | `external_transfer_svd_vf_paralysis_unified_main_20260315/svd_transfer_vf_paralysis/subject_predictions_tier1.csv` |
| `modma.depression` | 0 | few_shot | 42 | `followup_20260404/completed/external_transfer/modma_additional/modma_marvel_few_shot_depression_k5_seed0/few_shot_subject_predictions_tier2.csv` |
| `modma.depression` | 0 | zero_shot | 36 | `followup_20260404/completed/external_transfer/modma_additional/modma_marvel_zero_shot_depression/subject_predictions_tier2.csv` |