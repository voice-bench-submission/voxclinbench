# Split manifests — partial (test-set only)

6 manifests, one per (task, seed) extracted from the released prediction CSVs.

| Task | Seed | N test | Baselines contributing |
|---|---|---|---|
| `b2ai.tier1_aggregate` | 0 | 126 | fair (126 subjects) |
| `b2ai.tier2_aggregate` | 0 | 126 | fair (126 subjects) |
| `lodo.pd.test_neurovoz` | 0 | 108 | few_shot (98 subjects), main (108 subjects), zero_shot (98 subjects) |
| `lodo.pd.test_svd` | 0 | 32 | main (32 subjects), unified (32 subjects) |
| `lodo.vfp.test_svd` | 0 | 53 | main (53 subjects), unified (53 subjects) |
| `modma.depression` | 0 | 47 | few_shot (42 subjects), zero_shot (36 subjects) |

## How to reproduce train/val splits

Once the upstream corpus is fetched (`voxbench fetch <corpus>`), call:

```python
from voxbench.data import make_splits
split = make_splits(task_id="b2ai.parkinsons", seed=0)
# returns {"train": [...], "val": [...], "test": [...]}
```

The test-subject list returned should match this manifest's `test_subjects` field byte-for-byte.
