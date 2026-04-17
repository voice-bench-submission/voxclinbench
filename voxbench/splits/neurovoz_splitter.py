"""NeuroVoz deterministic subject-ID splitter (CC-BY-NC-ND-4.0 compliant).

NeuroVoz is released on Zenodo (record 10777657) under the CC-BY-NC-ND
4.0 license. The ND ("NoDerivatives") clause forbids redistributing
derivatives of the original work. A subject-ID split manifest CSV listing
our train / val / test splits is a derivative work. To avoid a license
violation, VoxClinBench does NOT ship a NeuroVoz split manifest; instead
we ship this deterministic splitter that regenerates the same 80/20
subject-stratified split locally from the user's own fetched copy.

Usage
-----
After fetching NeuroVoz locally (Zenodo 10777657) -- the canonical layout
has a top-level ``audios/`` directory containing per-subject WAV files
named ``PD-<id>-<suffix>.wav`` or ``HC-<id>-<suffix>.wav`` and a
metadata spreadsheet ``Metadata.xlsx`` -- run::

    python -m voxbench.splits.neurovoz_splitter \\
        --data_dir /path/to/neurovoz \\
        --seed 42 \\
        --out_dir /path/to/local/workspace/splits/

The script enumerates subject IDs (union of ``PD-<id>`` and ``HC-<id>``)
from the audio filenames, constructs the binary PD-vs-HC label vector,
and calls ``voxbench.data.make_splits`` with ``stratify=labels``. The
result is saved as ``neurovoz.parkinsons.seed<seed>.json`` in the
user-specified ``--out_dir`` (NOT in the voxbench repo). The content is
identical, byte-for-byte, to the manifest VoxClinBench-Base used for the
A9 leaderboard row.

Reviewer note: the determinism guarantee is "given the same fetched
NeuroVoz filesystem snapshot and the same seed, this script produces the
same split". That closes the reproducibility loop without redistributing
a derivative work.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Iterable

from voxbench.data.splits import Split, make_splits


logger = logging.getLogger(__name__)

# NeuroVoz audio filenames follow the pattern
#   "<PD|HC>-<subject_id>-<utterance_id>.wav"
# e.g. "PD-001-vowel_a.wav", "HC-042-sentence_03.wav".
_NEUROVOZ_FILENAME_RE = re.compile(r"^(PD|HC)-(\d+)-", re.IGNORECASE)

# Task identifier matching the A9 leaderboard row.
NEUROVOZ_TASK_ID = "neurovoz.parkinsons"


def _enumerate_subjects(data_dir: Path) -> list[tuple[str, int]]:
    """Walk NeuroVoz audio files and return sorted (subject_id, label) pairs.

    Label convention: 1 = PD, 0 = HC (matches the paper's A9 row
    direction: positives = Parkinson's disease).
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"NeuroVoz data_dir does not exist: {data_dir}. Fetch the "
            f"corpus from https://zenodo.org/record/10777657 first."
        )

    # Accept either a top-level audios/ directory or a flat layout.
    audio_roots = [data_dir / "audios", data_dir]
    audio_root = next((r for r in audio_roots if r.exists() and r.is_dir()), None)
    if audio_root is None:
        raise FileNotFoundError(
            f"No audio subdirectory found under {data_dir}; expected "
            f"either {data_dir / 'audios'} or WAV files directly in "
            f"{data_dir}."
        )

    subjects: dict[str, int] = {}
    for wav_path in sorted(audio_root.rglob("*.wav")):
        match = _NEUROVOZ_FILENAME_RE.match(wav_path.name)
        if match is None:
            logger.debug("Skipping non-matching filename: %s", wav_path.name)
            continue
        group, subject_num = match.group(1).upper(), match.group(2)
        subject_id = f"{group}-{subject_num}"
        label = 1 if group == "PD" else 0
        # Tolerate duplicate per-subject entries from multiple utterances,
        # but fail loud on label disagreements within a subject.
        if subject_id in subjects and subjects[subject_id] != label:
            raise ValueError(
                f"Subject {subject_id} has conflicting labels across "
                f"audio files (PD vs HC); this violates the NeuroVoz "
                f"schema. Path: {wav_path}"
            )
        subjects[subject_id] = label

    if not subjects:
        raise ValueError(
            f"Found no NeuroVoz-pattern WAV files under {audio_root}. "
            f"Expected filenames like PD-001-vowel_a.wav or "
            f"HC-042-sentence.wav."
        )

    # Sort by subject_id for determinism regardless of filesystem order.
    return sorted(subjects.items())


def regenerate_neurovoz_split(
    data_dir: Path,
    seed: int,
    test_frac: float = 0.20,
    val_frac_within_train: float = 0.20,
) -> Split:
    """Regenerate the A9 NeuroVoz-es Parkinson's split deterministically.

    Returns the :class:`voxbench.data.splits.Split` object; the caller
    decides whether to serialise it via :meth:`Split.as_manifest` into
    their own local workspace. **Do not write this split back into the
    voxbench repository** -- that would be a CC-BY-NC-ND violation.
    """
    pairs = _enumerate_subjects(data_dir)
    subjects = [sid for sid, _ in pairs]
    labels = [lbl for _, lbl in pairs]

    logger.info(
        "NeuroVoz: %d subjects (%d PD, %d HC) enumerated from %s",
        len(subjects),
        sum(labels),
        len(labels) - sum(labels),
        data_dir,
    )

    return make_splits(
        task_id=NEUROVOZ_TASK_ID,
        seed=seed,
        subjects=subjects,
        labels=labels,
        test_frac=test_frac,
        val_frac_within_train=val_frac_within_train,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the VoxClinBench A9 NeuroVoz-es Parkinson's "
            "subject-ID split locally (CC-BY-NC-ND-4.0 compliant; does "
            "not write inside the voxbench repository)."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help=(
            "Path to your locally-fetched NeuroVoz corpus "
            "(Zenodo record 10777657). Must contain an ``audios/`` "
            "directory or WAV files named like PD-001-*.wav / HC-042-*.wav."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split seed; the paper uses 42 for A9 (default: 42).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "OPTIONAL directory to write the regenerated split JSON to "
            "(in your local workspace only). If omitted, the manifest "
            "is printed to stdout so it can be piped elsewhere."
        ),
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.20,
        help="Outer-level test fraction (default: 0.20).",
    )
    parser.add_argument(
        "--val_frac_within_train",
        type=float,
        default=0.20,
        help="Validation fraction within the non-test set (default: 0.20).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit INFO-level logging.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    split = regenerate_neurovoz_split(
        data_dir=args.data_dir,
        seed=args.seed,
        test_frac=args.test_frac,
        val_frac_within_train=args.val_frac_within_train,
    )
    manifest = split.as_manifest()

    if args.out_dir is None:
        json.dump(manifest, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    out_path = args.out_dir / f"{NEUROVOZ_TASK_ID}.seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote regenerated NeuroVoz split to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
