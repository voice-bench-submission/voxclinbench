# Splits

Each file here is a JSON manifest for one (task, seed) combination:

```json
{
  "task_id": "b2ai.parkinsons",
  "seed": 42,
  "train_subjects": ["sub-0001", ...],
  "val_subjects":   ["sub-0123", ...],
  "test_subjects":  ["sub-0456", ...]
}
```

TODO: populate the 22 tasks x 3 seeds = 66 JSON manifests from
`artifacts/modal_sync/` and from the existing scripts/eval_*.py splits
at camera-ready.
