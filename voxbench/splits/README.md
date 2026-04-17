# Subject-ID split regenerators (Python package)

This sub-package holds script-based splitters for corpora whose license
does not permit redistributing derivative split manifests (e.g.
NeuroVoz under CC-BY-NC-ND-4.0). Each splitter reads only from the
user's locally-fetched copy of the corpus and returns a deterministic
split in-memory.

| Module | Task row | Corpus license | Status |
|---|---|---|---|
| `neurovoz_splitter` | A9 `neurovoz.parkinsons` | CC-BY-NC-ND-4.0 (Zenodo 10777657) | ND clause → script-only |

For corpora that permit manifest redistribution, the shipped JSON
manifests live one directory up at
[`voxbench/splits/`](../../splits/), built from
`scripts/build_manifests.py`; see [`splits/INDEX.md`](../../splits/INDEX.md)
for the current coverage and the reproducible train/val regeneration
path.

## Why two locations

- `voxbench/voxbench/splits/` (this package) — **Python code**: importable
  splitter modules. No split data.
- `voxbench/splits/` (repo top-level) — **data-only**: JSON subject-ID
  manifests shipped for corpora whose license allows redistribution.
