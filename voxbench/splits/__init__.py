"""Subject-ID split regenerators.

This package holds *script-based* splitters for corpora whose license
does not permit redistributing derivative manifests (e.g. NeuroVoz
under CC-BY-NC-ND-4.0). For corpora that allow manifest redistribution,
the shipped JSON files live one directory up at ``splits/``.

The splitters here do NOT read or write files inside the voxbench
repository; they read only from the user's local fetched corpus and
return a deterministic split in-memory (optionally writing to a user-
specified path inside the user's workspace).
"""
