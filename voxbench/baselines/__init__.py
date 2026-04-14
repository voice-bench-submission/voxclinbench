"""Reference baselines for VoxClinBench.

Two baselines are packaged:

- MARVEL: WavLM-Large mean-pooled + LogReg, frozen. The reference
  implementation ships alongside this package under
  `voxbench.baselines.marvel` (TODO). TODO: port as a minimal wrapper
  that consumes precomputed WavLM embeddings.

- VoxClinBench-Base: 8-modality CNN-Transformer with Griffin-Lim bridge.
  The reference training harness ships as `voxbench.train`; see the
  paper supplementary §B for the original ablation scripts. TODO: ship
  a minimal inference wrapper that loads the released state_dict and
  emits submission JSONs.

Both wrappers are TODOs; the scaffolding below defines the expected API
so the main `voxbench eval` entrypoint can dispatch to either baseline
without further changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class BaselineModel(Protocol):
    """Minimal Protocol every baseline implements."""

    name: str

    def predict_subject(self, subject_id: str) -> float:
        """Return probability that this subject has the task's condition."""
        ...


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    track: str  # "frozen" or "full_finetune"
    ref_script: str


SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        name="MARVEL",
        track="frozen",
        ref_script="scripts/eval_marvel.py",
    ),
    BaselineSpec(
        name="VoxClinBench-Base",
        track="frozen",
        ref_script="scripts/train_multimodal_b2ai.py",
    ),
)
