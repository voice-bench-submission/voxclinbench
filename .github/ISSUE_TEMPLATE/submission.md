---
name: Benchmark submission
about: Submit a new baseline / method to VoxClinBench
title: "[submission] <method name>"
labels: submission
---

## Method
One-paragraph description of your method (backbone, adaptation strategy, pooling, head).

## Which VoxClinBench tasks did you score?
- [ ] Family A — within-corpus (B2AI Tier 1/2 physical)
- [ ] Family A — NeuroVoz Spanish PD
- [ ] Family A — SVD German VFP
- [ ] Family B — within-corpus (B2AI + DAIC-WOZ + E-DAIC + MODMA depression/PTSD/ADHD)
- [ ] Family B — E-DAIC PHQ-8 regression (A16b)
- [ ] Cross-lingual LODO (B1--B6)

## Protocol compliance
- [ ] Subject-disjoint 80/20 train/test (we used the manifests in `splits/`).
- [ ] 5 random seeds (or state which subset you used).
- [ ] 2000 subject-level bootstrap resamples for 95 % CIs.
- [ ] Only our task's labeled train+val subjects were used for training (no test leakage).
- [ ] No per-subject demographic metadata or ground-truth labels were included in published artifacts.

## Submission artifacts
Please attach or link:
1. Per-seed per-subject probability CSVs (`subject_id, predicted_prob` only).
2. A config or command line reproducing your runs from an upstream checkpoint.
3. Per-seed scored AUROC/AUPRC (or Pearson r / CCC for A16b).

## Self-check against our DUA policy
- [ ] I did not redistribute raw audio from any upstream corpus.
- [ ] My released CSVs contain no labels, demographics, or PHI.
- [ ] My submission URL is reachable without personal request to me.
