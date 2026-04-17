---
name: Benchmark submission
about: Submit a new baseline / method to VoxClinBench
title: "[submission] <method name>"
labels: submission
---

## Method
One-paragraph description of your method (backbone, adaptation strategy, pooling, head).

## Which VoxClinBench tasks did you score?
### Family A (physical / motor / laryngeal / respiratory)
- [ ] Family A — within-Bridge2AI (B2AI Tier 1/2 physical: A1–A8)
- [ ] Family A — NeuroVoz Spanish PD (A9, external cohort; regenerate split via `voxbench.splits.neurovoz_splitter`)
- [ ] Family A — SVD German VFP (A10, external cohort)
- [ ] Family A — SVD German hyperfunctional dysphonia (A11)
- [ ] Family A — SVD German laryngitis (A12)
- [ ] Family A — SVD German functional dysphonia (A13)
- [ ] Family A — SVD German psychogenic dysphonia (A14)
- [ ] Family A — SVD German contact pachydermia (A15)
- [ ] Family A — SVD German Reinke's edema (A16)
- [ ] Family A — SVD German dysodia (A17)
- [ ] Family A — SVD German vocal fold polyp (A18)

### Family B (psychiatric)
- [ ] Family B — within-Bridge2AI (B2AI depression/PTSD/ADHD: A19–A21)
- [ ] Family B — E-DAIC English depression (A22, external cohort)
- [ ] Family B — E-DAIC English PTSD (A23, external cohort; PCL-C binary)

### Cross-lingual reference (not scored leaderboard rows)
- [ ] Zero-shot validation — MODMA Chinese depression (A24)

## Training-corpus disclosure (submission metadata)
Cross-lingual transfer is submission-disclosed, not a benchmark task.
Please state, per scored row, which corpora you trained on (e.g. "A9
scored zero-shot from B2AI-en only", "A10-A18 scored after B2AI+SVD
joint fine-tune", etc.). Zero-shot cross-lingual AUROC may be reported
as a descriptive finding; it does not alter the row's benchmark rank.

## Protocol compliance
- [ ] Subject-disjoint 80/20 train/test (we used the manifests in `splits/` where shipped, and regenerated locally via `voxbench.splits.neurovoz_splitter` for A9).
- [ ] 5 random seeds (or state which subset you used).
- [ ] 2000 subject-level bootstrap resamples for 95 % CIs.
- [ ] Only our task's labeled train+val subjects were used for training (no test leakage).
- [ ] No per-subject demographic metadata or ground-truth labels were included in published artifacts.

## Submission artifacts
Please attach or link:
1. Per-seed per-subject probability CSVs (`subject_id, predicted_prob` only).
2. A config or command line reproducing your runs from an upstream checkpoint.
3. Per-seed scored AUROC/AUPRC per task.

## Self-check against our DUA policy
- [ ] I did not redistribute raw audio from any upstream corpus.
- [ ] My released CSVs contain no labels, demographics, or PHI.
- [ ] My submission URL is reachable without personal request to me.
- [ ] For NeuroVoz (A9): I did NOT redistribute a split-manifest CSV (the splitter script regenerates it locally under CC-BY-NC-ND-4.0 ND compliance).
