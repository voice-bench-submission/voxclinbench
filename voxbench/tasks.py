"""VoxClinBench task registry.

18 live tasks (v0.2, post 2026-04-14 retraction):
- Family A (physical): 10 within-corpus (A1--A8, A13, A14)
- Family B (psychiatric): 6 within-corpus (A9--A11, A16a, A16b, A17)
- Cross-lingual LODO: 2 (B1 PD EN->ES on NeuroVoz, B6 VFP EN->DE on SVD)

Retracted 2026-04-14 (hard-deleted from registry; see
paper/voxbench_paper/snippets/task_card.tex and supp
\\S\\ref{sec:lodo-audit} for the retraction ledger):
- A15 daicwoz.depression: DAIC-WOZ subjects are a strict subset of E-DAIC
  (189/189 overlap), so within-DAIC-WOZ is not an independent task.
- B2 lodo.dep.test_b2ai, B3 lodo.dep.test_daic,
  B4 lodo.dep.test_edaic, B5 lodo.dep.test_modma: the depression LODO
  family was contaminated by the DAIC-WOZ subset of E-DAIC and by target
  corpora appearing in the train set. Superseded by A9, A16a, A17
  within-corpus results.

B2AI's "prior psychiatric history" label (A12) is intentionally excluded
from the benchmark: it is a composite self-report that overlaps
depression, PTSD, and ADHD in the same cohort and therefore does not
define an independent biomarker task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Family = Literal["A", "B"]
TaskKind = Literal["within", "lodo"]


@dataclass(frozen=True)
class Task:
    """Immutable task definition.

    A Task is one row of the VoxClinBench leaderboard. It binds a task_id
    to its corpus, language, disease, evaluation metric, and split
    protocol. Split manifests (subject IDs per seed) live in
    voxbench/splits/<task_id>.json (TODO: populate from artifacts/).
    """

    task_id: str
    family: Family
    kind: TaskKind
    corpus: str
    language: str
    condition: str
    primary_metric: str
    secondary_metric: str | None
    n_pos: int | None
    n_neg: int | None
    notes: str = ""
    # LODO-only: corpora trained on before evaluating on this task's corpus
    lodo_train_corpora: tuple[str, ...] = field(default_factory=tuple)


TASKS: tuple[Task, ...] = (
    # --- Family A: physical / motor / laryngeal / respiratory (within-corpus) ---
    Task("b2ai.parkinsons",          "A", "within", "bridge2ai", "en",
         "Parkinson's disease",        "AUROC", "AUPRC", 73,  689),
    Task("b2ai.airway_stenosis",     "A", "within", "bridge2ai", "en",
         "iSGS airway stenosis",       "AUROC", "AUPRC", 55,  707),
    Task("b2ai.vf_paralysis",        "A", "within", "bridge2ai", "en",
         "Vocal fold paralysis",       "AUROC", "AUPRC", 60,  702),
    Task("b2ai.laryngeal_dystonia",  "A", "within", "bridge2ai", "en",
         "Adductor spasmodic dysphonia","AUROC", "AUPRC", 50, 712),
    Task("b2ai.cognitive_impair",    "A", "within", "bridge2ai", "en",
         "Cognitive impairment",       "AUROC", "AUPRC", 45,  717),
    Task("b2ai.benign_lesions",      "A", "within", "bridge2ai", "en",
         "Benign vocal-fold lesions",  "AUROC", "AUPRC", 58,  704),
    Task("b2ai.mtd",                 "A", "within", "bridge2ai", "en",
         "Muscle tension dysphonia",   "AUROC", "AUPRC", 52,  710),
    Task("b2ai.chronic_cough",       "A", "within", "bridge2ai", "en",
         "Chronic cough",              "AUROC", "AUPRC", 41,  721),
    Task("neurovoz.parkinsons",      "A", "within", "neurovoz",  "es",
         "Parkinson's disease",        "AUROC", None,    53,   54),
    Task("svd.vf_paralysis",         "A", "within", "svd",       "de",
         "Vocal fold paralysis",       "AUROC", None,    213,  300),
    # --- Family B: psychiatric (within-corpus) ---
    Task("b2ai.depression",          "B", "within", "bridge2ai", "en",
         "Depression",                 "AUROC", "AUPRC", 44,   718),
    Task("b2ai.ptsd",                "B", "within", "bridge2ai", "en",
         "PTSD",                       "AUROC", "AUPRC", 38,   724),
    Task("b2ai.adhd",                "B", "within", "bridge2ai", "en",
         "ADHD",                       "AUROC", "AUPRC", 47,   715),
    Task("edaic.depression",         "B", "within", "edaic",     "en",
         "Depression (PHQ-8>=10)",     "AUROC", None,    82,   193),
    Task("edaic.phq8_regression",    "B", "within", "edaic",     "en",
         "PHQ-8 score regression",     "Pearson_r", "CCC", None, None,
         notes="Baseline pending (see paper supplementary §A16b)."),
    Task("modma.depression",         "B", "within", "modma",     "zh",
         "MDD vs healthy control",     "AUROC", None,    23,   29),
    # --- Cross-lingual LODO ---
    Task("lodo.pd.test_neurovoz",    "A", "lodo", "neurovoz", "es",
         "Parkinson's disease (transfer)", "AUROC", None, None, None,
         lodo_train_corpora=("bridge2ai",)),
    Task("lodo.vfp.test_svd",        "A", "lodo", "svd", "de",
         "Vocal fold paralysis (transfer)", "AUROC", None, None, None,
         lodo_train_corpora=("bridge2ai",),
         notes="Reveals Griffin-Lim cross-domain gap; see paper §6(c)."),
)


_BY_ID = {t.task_id: t for t in TASKS}


def get_task(task_id: str) -> Task:
    """Return the Task with this ID, or raise KeyError."""
    if task_id not in _BY_ID:
        raise KeyError(f"Unknown VoxClinBench task: {task_id!r}")
    return _BY_ID[task_id]


def list_task_ids(family: Family | None = None,
                  kind: TaskKind | None = None) -> tuple[str, ...]:
    """List task IDs, optionally filtered by family or kind."""
    def _keep(t: Task) -> bool:
        if family is not None and t.family != family:
            return False
        if kind is not None and t.kind != kind:
            return False
        return True

    return tuple(t.task_id for t in TASKS if _keep(t))
