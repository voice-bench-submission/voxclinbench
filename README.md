# VoxClinBench

First cross-lingual, cross-disease clinical voice biomarker benchmark.
The benchmark pairs a voice-generalist leaderboard with a language-shift
falsification test, so that accuracy on a single corpus cannot be
mistaken for cross-population generalization.
Five corpora (Bridge2AI-Voice v3.0, NeuroVoz, SVD, E-DAIC, MODMA),
four languages (en, es, de, zh), 18 live tasks (16 within-corpus + 2
cross-lingual LODO) split into two sub-leaderboards (Family A:
physical/motor/laryngeal/respiratory; Family B: psychiatric).
DAIC-WOZ was retired (189/189 subset of E-DAIC); see CHANGELOG.

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
svd            CC BY 4.0 Zenodo mirror (publicly downloadable)
neurovoz       CC BY-NC-ND 4.0 (Zenodo access request)
modma          CC BY-NC 4.0 (Lanzhou University form)
```

## Task list

18 live tasks defined in `voxbench/tasks.py` (A15 daicwoz.depression and
B2-B5 depression LODO retracted after the 189/189 DAIC-WOZ subset audit;
see CHANGELOG). The paper's supplementary §A gives the full catalogue
with task-level statistics.

| Family | Within-corpus | LODO | Example conditions |
|---|---|---|---|
| A (physical) | 10 | 2 | PD (en/es), VFP (en/de), iSGS, laryngeal dystonia, MCI, benign lesions, MTD, chronic cough |
| B (psychiatric) | 6 | 0 | depression (B2AI, E-DAIC, MODMA zero-shot target), PTSD, ADHD, PHQ-8 regression |

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
