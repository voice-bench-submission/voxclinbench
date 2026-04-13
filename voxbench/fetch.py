"""Download-on-use CLI for VoxClinBench source corpora.

Access gates (most to least restrictive):
- bridge2ai:       PhysioNet credentialed; hard login wall; DUA required.
- modma:           Lanzhou University application form (CC BY-NC 4.0).
- neurovoz:        CC BY-NC-ND 4.0; Zenodo access request (files restricted).
- daicwoz, edaic:  USC/ICT EULA governs USE; files HTTP-reachable at
                   dcapswoz.ict.usc.edu without login — the EULA at
                   the same site auto-applies on download.
- svd:             CC BY 4.0 Zenodo mirror (records 16874898 + 7024894);
                   files publicly downloadable without any gate.

This module intentionally does NOT bundle raw audio: each provider has
its own redistribution policy, and respecting that boundary is what
makes VoxClinBench safe to submit under each corpus's DUA. The
function below validates credential env vars (when applicable) and
prints the official URL plus target path; the actual HTTP download
per corpus is tracked as v0.3 work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusSource:
    name: str
    license: str
    access: str
    url: str
    credential_env: str | None = None


SOURCES: dict[str, CorpusSource] = {
    "bridge2ai": CorpusSource(
        name="Bridge2AI-Voice v3.0",
        license="PhysioNet Credentialed 1.5.0",
        access="PhysioNet credentialing + signed DUA",
        url="https://physionet.org/content/bridge2ai-voice/3.0.0/",
        credential_env="PHYSIONET_TOKEN",
    ),
    "neurovoz": CorpusSource(
        name="NeuroVoz",
        license="CC BY-NC-ND 4.0",
        access="Zenodo access request (files are restricted)",
        url="https://doi.org/10.5281/zenodo.10777657",
    ),
    "svd": CorpusSource(
        name="Saarbruecken Voice Database",
        license="CC BY 4.0 (Zenodo mirror)",
        access="public",
        url="https://doi.org/10.5281/zenodo.16874898",
    ),
    "daicwoz": CorpusSource(
        name="DAIC-WOZ",
        license="USC/ICT EULA (use governed; files HTTP-reachable)",
        access="EULA auto-applies on download from dcapswoz.ict.usc.edu",
        url="https://dcapswoz.ict.usc.edu/wwwdaicwoz/",
        credential_env="USC_EULA_ACCEPTED",
    ),
    "edaic": CorpusSource(
        name="E-DAIC (AVEC'19)",
        license="USC/ICT EULA (use governed; files HTTP-reachable)",
        access="EULA auto-applies on download from dcapswoz.ict.usc.edu",
        url="https://dcapswoz.ict.usc.edu/wwwedaic/",
        credential_env="USC_EULA_ACCEPTED",
    ),
    "modma": CorpusSource(
        name="MODMA",
        license="CC BY-NC 4.0",
        access="Lanzhou University application form",
        url="http://modma.lzu.edu.cn/data/index/",
    ),
}


def check_credentials(source: CorpusSource) -> None:
    """Raise PermissionError if required credentials are not present."""
    if source.credential_env and not os.environ.get(source.credential_env):
        raise PermissionError(
            f"{source.name} requires {source.access}. "
            f"Set env var {source.credential_env} after completing "
            f"credentialing / EULA."
        )


def fetch(dataset: str, target: Path | str | None = None) -> Path:
    """Document the fetch flow for a named corpus.

    This function validates your credentials (where applicable) and
    prints the official upstream URL plus the target path where you
    should place the downloaded audio. It deliberately does NOT bundle
    or proxy the audio: each upstream provider runs its own credential
    flow (PhysioNet DUA, USC/ICT EULA, Saarland form, Lanzhou form, or
    CC-licensed Zenodo for NeuroVoz) and VoxClinBench respects that
    boundary rather than caching files on our side.

    For the open-licensed NeuroVoz corpus a one-shot downloader is a
    natural extension and is tracked as a v0.3 follow-up (the
    canonical Zenodo record and file naming moved between releases;
    we intentionally do not hard-code a stale URL here).
    """
    if dataset not in SOURCES:
        raise KeyError(
            f"Unknown dataset {dataset!r}. Known: {sorted(SOURCES)}"
        )
    source = SOURCES[dataset]
    check_credentials(source)
    target_path = Path(target or Path.home() / ".voxbench" / dataset)
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"[voxbench] {source.name} ({source.license})")
    print(f"[voxbench] Access:  {source.access}")
    print(f"[voxbench] Source:  {source.url}")
    print(f"[voxbench] Target:  {target_path}")
    print("[voxbench] Per-corpus downloaders are tracked as v0.3 work; "
          "download audio from the URL above into the target path once "
          "your credentials are approved.")
    return target_path
