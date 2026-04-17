"""VoxClinBench task registry.

24 total registrations in v0.4.1 (2026-04-16):
- 23 scored benchmark tasks:
  - Family A (physical / motor / laryngeal / respiratory): 18 tasks
    - 8 within-Bridge2AI (A1-A8)
    - 1 external cohort NeuroVoz-es Parkinson's (A9)
    - 9 external cohort SVD-de pathologies (A10 vf_paralysis,
      A11 hyperfunctional_dysphonia, A12 laryngitis,
      A13 functional_dysphonia, A14 psychogenic_dysphonia,
      A15 contact_pachydermia, A16 reinke_edema, A17 dysodia,
      A18 vf_polyp)
  - Family B (psychiatric): 5 tasks
    - 3 within-Bridge2AI (A19 depression, A20 PTSD, A21 ADHD)
    - 2 external cohort E-DAIC-en (A22 depression, A23 PTSD binary)
- 1 zero-shot cross-lingual reference row (not scored):
  - A24 modma.depression (Mandarin MDD vs HC; outside the Holm family)

Use ``list_scorable_tasks()`` for the 23-row leaderboard view. The
A24 cross-lingual reference row is filtered out of the scored list.

Cross-lingual transfer is NOT a benchmark task category. A task specifies
(data, labels, metric); which corpora a submission trains on is
submission-disclosed metadata, not a benchmark-prescribed constraint.
Zero-shot cross-lingual AUROC (e.g. B2AI-en training -> NeuroVoz-es test)
may be reported by submitters as a descriptive finding; the benchmark
rank target is the within-cohort AUROC on the external-cohort task row
(A9 NeuroVoz-es, A10-A18 SVD-de, A22-A23 E-DAIC-en). The A24 MODMA-zh
row is a cross-lingual reference, not a scored rank.

Task numbering resequenced in v0.4.0 and updated in v0.4.1. In v0.4.1
the MODMA row was renumbered A26 -> A24, and the former A24/A25
regression tasks (edaic.phq8_regression, edaic.ptsd_severity) were
retired from the scored registry. The former A12 / A15 / A16a / A16b
/ A17 suffix notation is retired; see the CHANGELOG for the full
retraction ledger.

SVD pathology tasks use the German pathology folder names from the
Saarland SVD release (Zenodo record 16874898, CC-BY-4.0):
- A10 vf_paralysis           <- Stimmlippenlähmung
- A11 hyperfunctional_dysphonia <- Hyperfunktionelle Dysphonie
- A12 laryngitis             <- Laryngitis
- A13 functional_dysphonia   <- Funktionelle Dysphonie
- A14 psychogenic_dysphonia  <- Psychogene Dysphonie
- A15 contact_pachydermia    <- Kontaktpachydermie
- A16 reinke_edema           <- Reinke Ödem
- A17 dysodia                <- Dysodie
- A18 vf_polyp               <- Stimmlippenpolyp
All SVD tasks share the same n_healthy_subjects=300 pool under the
v0.4.x release (XLSR-53 L18 frozen + LogReg baseline, 5 seeds,
subject-stratified 80/20 split).

E-DAIC PTSD binary labels (A23) derive from the PCL-C questionnaire
column in ``detailed_lables.csv``: ``PTSD_label`` (binary, cutoff-based).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Family = Literal["A", "B"]
TaskKind = Literal["within", "external"]


@dataclass(frozen=True)
class Task:
    """Immutable task definition.

    A Task is one row of the VoxClinBench leaderboard. It binds a task_id
    to its corpus, language, disease, evaluation metric, and split
    protocol. Split manifests (subject IDs per seed) live in
    voxbench/splits/<task_id>.json. TaskKind is either ``within``
    (subject-disjoint 80/20 split on a single corpus) or ``external``
    (entire external cohort serves as the test set; training-corpus
    choice is submission-disclosed and not prescribed by the benchmark).
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


TASKS: tuple[Task, ...] = (
    # --- Family A: physical / motor / laryngeal / respiratory ---
    # A1-A8: Within-Bridge2AI
    Task("b2ai.parkinsons",          "A", "within",   "bridge2ai", "en",
         "Parkinson's disease",        "AUROC", "AUPRC", 73,  689),
    Task("b2ai.airway_stenosis",     "A", "within",   "bridge2ai", "en",
         "iSGS airway stenosis",       "AUROC", "AUPRC", 55,  707),
    Task("b2ai.vf_paralysis",        "A", "within",   "bridge2ai", "en",
         "Vocal fold paralysis",       "AUROC", "AUPRC", 60,  702),
    Task("b2ai.laryngeal_dystonia",  "A", "within",   "bridge2ai", "en",
         "Adductor spasmodic dysphonia","AUROC", "AUPRC", 50, 712),
    Task("b2ai.cognitive_impair",    "A", "within",   "bridge2ai", "en",
         "Cognitive impairment",       "AUROC", "AUPRC", 45,  717),
    Task("b2ai.benign_lesions",      "A", "within",   "bridge2ai", "en",
         "Benign vocal-fold lesions",  "AUROC", "AUPRC", 58,  704),
    Task("b2ai.mtd",                 "A", "within",   "bridge2ai", "en",
         "Muscle tension dysphonia",   "AUROC", "AUPRC", 52,  710),
    Task("b2ai.chronic_cough",       "A", "within",   "bridge2ai", "en",
         "Chronic cough",              "AUROC", "AUPRC", 41,  721),
    # A9: NeuroVoz external cohort
    Task("neurovoz.parkinsons",      "A", "external", "neurovoz",  "es",
         "Parkinson's disease",        "AUROC", None,    53,   55),
    # A10-A18: SVD external cohort pathologies (CC-BY-4.0, Zenodo 16874898)
    Task("svd.vf_paralysis",                "A", "external", "svd", "de",
         "Vocal fold paralysis",       "AUROC", None,    213,  300,
         notes="SVD pathology: Stimmlippenlähmung."),
    Task("svd.hyperfunctional_dysphonia",   "A", "external", "svd", "de",
         "Hyperfunctional dysphonia",  "AUROC", None,    199,  300,
         notes="SVD pathology: Hyperfunktionelle Dysphonie."),
    Task("svd.laryngitis",                  "A", "external", "svd", "de",
         "Laryngitis",                 "AUROC", None,    128,  300,
         notes="SVD pathology: Laryngitis."),
    Task("svd.functional_dysphonia",        "A", "external", "svd", "de",
         "Functional dysphonia",       "AUROC", None,    108,  300,
         notes="SVD pathology: Funktionelle Dysphonie."),
    Task("svd.psychogenic_dysphonia",       "A", "external", "svd", "de",
         "Psychogenic dysphonia",      "AUROC", None,    80,   300,
         notes="SVD pathology: Psychogene Dysphonie."),
    Task("svd.contact_pachydermia",         "A", "external", "svd", "de",
         "Contact pachydermia",        "AUROC", None,    63,   300,
         notes="SVD pathology: Kontaktpachydermie."),
    Task("svd.reinke_edema",                "A", "external", "svd", "de",
         "Reinke's edema",             "AUROC", None,    54,   300,
         notes="SVD pathology: Reinke Ödem."),
    Task("svd.dysodia",                     "A", "external", "svd", "de",
         "Dysodia (singing voice disorder)", "AUROC", None, 54, 300,
         notes="SVD pathology: Dysodie."),
    Task("svd.vf_polyp",                    "A", "external", "svd", "de",
         "Vocal fold polyp",           "AUROC", None,    40,   300,
         notes="SVD pathology: Stimmlippenpolyp."),
    # --- Family B: psychiatric ---
    # A19-A21: Within-Bridge2AI
    Task("b2ai.depression",          "B", "within",   "bridge2ai", "en",
         "Depression",                 "AUROC", "AUPRC", 44,   718),
    Task("b2ai.ptsd",                "B", "within",   "bridge2ai", "en",
         "PTSD",                       "AUROC", "AUPRC", 38,   724),
    Task("b2ai.adhd",                "B", "within",   "bridge2ai", "en",
         "ADHD",                       "AUROC", "AUPRC", 47,   715),
    # A22-A23: E-DAIC external cohort (English)
    Task("edaic.depression",         "B", "external", "edaic",     "en",
         "Depression (PHQ-8>=10)",     "AUROC", None,    82,   193),
    Task("edaic.ptsd",               "B", "external", "edaic",     "en",
         "PTSD (PCL-C cutoff)",        "AUROC", None,    87,   188,
         notes="E-DAIC PTSD binary from PCL-C cutoff; label column "
               "``PTSD_label`` in detailed_lables.csv (derived from "
               "PCL-C total score)."),
    # A24: Cross-lingual reference row (zero-shot only, not scored)
    Task("modma.depression",         "B", "external", "modma",     "zh",
         "MDD vs healthy control",     "AUROC", None,    23,   29,
         notes="Cross-lingual reference row (zero-shot only, n=52). "
               "Outside the Holm family; directional cross-lingual "
               "reference, not a scored ranking."),
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


# Task IDs held in the registry for reproducibility but NOT counted as
# scored leaderboard rows. Currently:
# - A24 ``modma.depression`` (cross-lingual zero-shot reference row;
#   outside the Holm family, not a scored rank)
_NON_SCORED_TASK_IDS = frozenset({
    "modma.depression",
})

# Back-compat alias for v0.4.0 consumers that imported the old name.
_SCOPING_TASK_IDS = _NON_SCORED_TASK_IDS


def list_scorable_tasks() -> tuple[Task, ...]:
    """Return the 23 scored VoxClinBench tasks (excludes the cross-lingual
    reference row).

    The non-scored rows are kept in ``TASKS`` for reproducibility but do
    not define leaderboard rankings:
    - ``modma.depression`` (A24, zero-shot cross-lingual reference row;
      see module docstring for rationale).
    """
    return tuple(t for t in TASKS if t.task_id not in _NON_SCORED_TASK_IDS)
