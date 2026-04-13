# Changelog

## 0.2.0-dev

- Replaced bootstrap-based `delong_p` placeholder with a proper DeLong
  U-statistic kernel following Sun & Xu 2014 (IEEE SPL), "Fast
  Implementation of DeLong's Algorithm for Comparing the Areas under
  Correlated Receiver Operating Characteristic Curves". Uses mid-rank
  transformation for O(N log N) covariance estimation.
- `delong_p` now returns `(auc_diff, p_value)` and validates inputs
  (NaN, single-class labels, shape mismatch).
- Added `paired_bootstrap_p` as a non-parametric sanity check.
- `voxbench compare` accepts `--test delong` (default) and
  `--test paired-bootstrap`.
- Added `voxbench/tests/test_eval.py` with unit tests for identical
  predictors, significantly different predictors, edge cases, and
  sklearn cross-check.

## 0.1.0

- Initial release.
