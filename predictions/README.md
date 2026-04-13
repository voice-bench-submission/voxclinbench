# Reference-baseline predictions

Per-subject prediction probabilities from the four reference baselines
shipped with VoxClinBench. Each CSV has exactly two columns:

```csv
subject_id,predicted_prob
```

**Labels, demographics, and ground-truth scores are NOT included.**
Reviewers with credentialed access to the upstream corpora can
reconstruct the full evaluation table by joining these predictions
with the upstream labels and running `voxbench eval`.

## Filename convention

```
<task_id>.seed<s>.<baseline>.csv
```

where `baseline` is one of: `main`, `unified`, `fair`, `zero_shot`,
`few_shot`, `adapted`. See `INDEX.md` for the full mapping of file
back to source experiment artifact.

## Coverage

This release (v0.2) includes predictions from the experiments where
subject-level outputs were saved. Tasks whose source scripts saved
only aggregate AUROC (no per-subject probabilities) are listed in
`INDEX.md` as "aggregate-only" and will be regenerated from the
archived Modal checkpoints in a v0.3 follow-up commit.

## How these files were produced

`scripts/build_release.py` scans the source experiment artifact tree
and produces the canonical two-column CSVs with an explicit column
allow-list (`subject_id`, `predicted_prob`) and a forbidden-column
deny-list (`target`, `label`, `diagnosis`, `phq8`, `age`, `sex`,
`gender`, `race`, `ethnicity`, `subset`, `site`, `cohort`, `split`).
Any row that does not pass both filters is dropped.
