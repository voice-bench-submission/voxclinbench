<!--
Thank you for contributing to VoxClinBench. Please confirm the
following before requesting review.
-->

## What this PR does

<!-- One paragraph. Say if it's a bug fix, new baseline, docs, or protocol change. -->

## Scope

- [ ] Docs / README only
- [ ] Harness code (eval / compare / fetch / submit)
- [ ] New reference baseline
- [ ] New task or split protocol (requires explicit discussion first — open an issue)
- [ ] CI / packaging

## Checklist

- [ ] Running `pytest` locally passes.
- [ ] No forbidden columns (label / target / diagnosis / phq8 / age / sex / gender / race / ethnicity) are introduced into any shipped CSV.
- [ ] No raw audio or audio-reconstructible features are committed.
- [ ] If this PR adds a baseline, I have included per-seed prediction CSVs (subject_id, predicted_prob columns only) and the build/config needed to reproduce them from an upstream checkpoint.
- [ ] If this PR changes scoring behaviour, I have updated the DeLong / paired-bootstrap regression tests in `tests/` to cover the change.
- [ ] If this PR changes the task registry or split protocol, I have bumped the task-registry semver and opened a discussion issue first.

## Related issue(s)

Closes #... (if applicable)
