# VoxClinBench

First cross-lingual, cross-disease clinical voice biomarker benchmark.
Six corpora (Bridge2AI-Voice v3.0, NeuroVoz, SVD, DAIC-WOZ, E-DAIC, MODMA),
four languages (en, es, de, zh), 22 tasks split into two sub-leaderboards
(Family A: physical/motor/laryngeal/respiratory; Family B: psychiatric).

## Quickstart

```bash
pip install -e .

voxbench fetch bridge2ai        # PhysioNet credentialing required
voxbench fetch daicwoz          # USC/ICT EULA required
voxbench fetch neurovoz         # CC BY 4.0, no gate
voxbench fetch modma            # CC BY-NC 4.0 form
voxbench fetch svd              # Saarland access form
voxbench fetch edaic            # USC/ICT EULA (AVEC'19)

voxbench eval --task b2ai.parkinsons --predictions my_preds.json
voxbench eval --all --predictions-dir ./runs/ --out leaderboard_row.json
voxbench compare --a baseline.json --b mine.json --test delong
```

## Task list

22 tasks defined in `voxbench/tasks.py`; see `paper/voxbench_spec.md` in
the parent repo for the full catalogue.

| Family | Within-corpus | LODO | Example conditions |
|---|---|---|---|
| A (physical) | 10 | 2 | PD (en/es), VFP (en/de), iSGS, laryngeal dystonia, MCI, benign lesions, MTD, chronic cough |
| B (psychiatric) | 7 | 3 | depression (en/zh), PTSD, ADHD, PHQ-8 regression |

B2AI's composite "prior psychiatric history" label is intentionally
excluded: it overlaps depression/PTSD/ADHD in the same cohort and is not
an independent biomarker task.

## Protocol

- Subject-disjoint stratified 80/20 train/test; internal 80/20 train/val.
- Three split seeds: `{13, 42, 2026}`.
- Primary metric: AUROC (Pearson r + CCC for A16b regression).
- Secondary: AUPRC for tasks with positive prevalence <15%.
- Bootstrap: 2000 resamples for 95% CIs.
- Between-submission: DeLong on per-subject probabilities.
- Correction: Holm-Bonferroni at alpha=0.05 within each family.

## License

- Code: MIT (see `LICENSE`)
- Splits, docs: CC BY 4.0
- Per-dataset access inherited (see `voxbench/fetch.py`).

## Citation

See `CITATION.cff` for the preferred citation.
