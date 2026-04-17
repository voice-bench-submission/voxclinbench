# VoxClinBench

**v0.4.1 · 2026-04-16** — 23 scored + 1 cross-lingual reference, 5 corpora, 4 languages.

Cross-lingual, cross-disease clinical voice biomarker benchmark. The
benchmark pairs a voice-generalist leaderboard with external-cohort
rows in non-English languages, so that accuracy on a single corpus
is not mistaken for cross-population generalization.
Five corpora (Bridge2AI-Voice v3.0, NeuroVoz, SVD, E-DAIC, MODMA),
four languages (en, es, de, zh), **24 tasks total**: 23 scored
(11 within-Bridge2AI + 12 external-cohort) split into two
sub-leaderboards (Family A: physical / motor / laryngeal / respiratory,
n=18; Family B: psychiatric, n=5), plus 1 zero-shot cross-lingual
reference row (A24 MODMA Mandarin depression; outside the Holm family).
DAIC-WOZ was retired (189/189 subset of E-DAIC) and the composite A12
``psychiatric_history`` self-report was excluded from scored tasks
because it overlaps A19-A21 in the same cohort; see [CHANGELOG](CHANGELOG.md#040-2026-04-16).

## Two workflows

**A. Submit your own model (most users).** Run your model with your
own preprocessing pipeline, write a two-column
``subject_id,predicted_prob`` CSV per (task, seed), and score it:

```bash
pip install voxbench          # pure numpy + scipy + sklearn, no torch

voxbench fetch <corpus>       # print official URL + target path

# ... run YOUR model, write predictions/b2ai.parkinsons.seed0.mine.csv ...

voxbench eval --task b2ai.parkinsons \
    --predictions predictions/b2ai.parkinsons.seed0.mine.csv \
    --labels your_b2ai_labels.csv
voxbench compare --a baseline.json --b mine.json --test delong
```

Voxbench does **not** care how you produced the predictions — no
requirement to use our preprocessing, our SSL probe, or our
architecture.

**B. Reproduce / extend our VoxClinBench-Base baseline.** If you want
to retrain the 7-branch CNN-Transformer we report in the paper
(``VoxClinBench-Base`` column of Table 2), install the full
training stack and invoke our Modal harness:

```bash
pip install "voxbench[train]"   # torch + transformers + librosa + h5py + modal
python -m voxbench.train --help
```

The preprocessing recipe (mel / MFCC / PPG / EMA / prosody / static /
SSL / clinical features → HDF5) is in ``voxbench.data.features`` and
``voxbench.data.dataset``. Running it requires a Modal credential
and the credentialed upstream corpus.

## fetch corpus table

```
bridge2ai      PhysioNet credentialed (hard login wall)
edaic          USC/ICT EULA (AVEC'19; subsumes the retired DAIC-WOZ cohort)
svd            CC BY 4.0 Zenodo mirror (publicly downloadable; record 16874898)
neurovoz       CC BY-NC-ND 4.0 (Zenodo access request, record 10777657)
modma          CC BY-NC 4.0 (Lanzhou University form)
```

### NeuroVoz split regeneration (CC-BY-NC-ND-4.0 ND clause)

NeuroVoz is released under CC-BY-NC-ND-4.0. The ND clause forbids
redistributing derivative works — a subject-ID split manifest CSV listing
our train/test subject IDs is a derivative work and therefore cannot be
redistributed. v0.4.0 replaces the redistributed CSV with a deterministic
splitter script:

```bash
# After fetching NeuroVoz locally from Zenodo record 10777657 under the
# CC-BY-NC-ND-4.0 license:
python -m voxbench.splits.neurovoz_splitter \
    --data_dir /path/to/neurovoz \
    --seed 42 \
    --out_dir ./splits/neurovoz.parkinsons/
```

The script emits a split manifest in-memory (or to a local path inside
the user's own NeuroVoz workspace); it does NOT write to the voxbench
repository. See `voxbench/splits/neurovoz_splitter.py`.

## Task list

23 scored tasks defined in [`voxbench/tasks.py`](voxbench/voxbench/tasks.py)
plus 1 zero-shot cross-lingual reference row (A24 MODMA Mandarin
depression; outside the Holm family).
See [CHANGELOG v0.4.1](CHANGELOG.md#041-2026-04-16) for the retraction
ledger (old A12, old A15, LODO B1-B6, svd.leukoplakia, svd.presbyphonia,
retired A24/A25 regression tasks).
The paper's supplementary §A gives the full catalogue with task-level
statistics.

| Family | Within-Bridge2AI | External cohort | Example conditions |
|---|---|---|---|
| A (physical, n=18 scored) | 8 (A1-A8) | 10 external-cohort tasks in Family A (A9 NeuroVoz-es PD; A10-A18 nine SVD-de pathologies) | PD (en/es), VFP (en/de), iSGS, laryngeal dystonia, MCI, benign lesions, MTD, chronic cough, SVD pathologies: hyperfunctional / functional / psychogenic dysphonia, laryngitis, contact pachydermia, Reinke's edema, dysodia, vocal fold polyp |
| B (psychiatric, n=5 scored) | 3 (A19-A21) | 2 (A22 E-DAIC-en depression + A23 E-DAIC-en PTSD) | depression (B2AI + E-DAIC), PTSD (B2AI + E-DAIC), ADHD (B2AI) |

Cross-lingual reference row (outside the Holm family; not scored):
- **A24** `modma.depression` (MODMA Mandarin MDD vs HC; zero-shot
  cross-lingual validation target, n=52, outside the Holm family)

### Why A12 `psychiatric_history` is not a scored task

The B2AI v3.0 `psychiatric_history` column is a composite self-report
label whose positive set is the union of the depression, PTSD, ADHD,
anxiety, and bipolar self-reports in the same cohort. It therefore
overlaps A19 depression + A20 PTSD + A21 ADHD by construction; its
signal is a noisy union of those three, not an independent biomarker.
The VoxClinBench-Base AUROC on this label (0.708 in earlier drafts)
is arithmetically consistent with a union of the A19-A21 signals and
adds no information beyond them. We therefore exclude A12 from the
scored task list and from the Family B psychiatric macro, which is now
``mean{A19 depression, A20 PTSD, A21 ADHD}`` (n=3). The label column
remains in the upstream B2AI raw-data tsv and in the training-side
``voxbench/config.py`` (`DISEASE_LIST`, `TIER_DISEASES[2]`) because
our reference pipeline keeps it as an auxiliary multi-task head for
regularization; it simply does not define a leaderboard row.

### Cross-lingual transfer is submission-disclosed, not a benchmark task

A task specifies (data, labels, metric). Which corpora a submission trains
on -- English-only, multilingual pool, target-corpus fine-tune -- is
submission-disclosed metadata, not a benchmark-prescribed constraint.
Submitters may report zero-shot cross-lingual AUROC (e.g. B2AI-en training
evaluated on NeuroVoz-es, SVD-de, or MODMA-zh) as a descriptive finding;
it is not a benchmark rank target. The benchmark scores each external
cohort as a within-cohort row (A9 NeuroVoz-es, A10-A18 SVD-de, A22-A23
E-DAIC-en, and A24 MODMA-zh as the zero-shot validation target)
regardless of training-data choice.

## Protocol

- Subject-disjoint stratified 80/20 train/test; internal 80/20 train/val.
- Five split seeds: `{42, 1, 2, 3, 4}` (see CHANGELOG for seed policy).
- Primary metric: AUROC.
- Secondary: AUPRC for tasks with positive prevalence <15%.
- Bootstrap: 2000 resamples for 95% CIs.
- Between-submission: DeLong on per-subject probabilities.
- Correction: Holm-Bonferroni at alpha=0.05 within each family.

## License

- Code: MIT (see `LICENSE`)
- Splits, docs: CC BY 4.0
- Per-dataset access inherited (see `voxbench/fetch.py`). NeuroVoz
  manifest is regenerated locally via
  `voxbench/splits/neurovoz_splitter.py` to respect the
  CC-BY-NC-ND-4.0 ND clause.

## Citation

See `CITATION.cff` for the preferred citation.
