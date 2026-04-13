"""
Miscellaneous training utilities: run logging and file cleanup.
"""
import os
import sys
from datetime import datetime


def _setup_run_logging(
    kind: str,
    tier: int | None = None,
    base_dir: str | None = None,
) -> str:
    """Tee stdout / stderr to a timestamped log file on the Modal volume.

    Returns the path of the created log file.
    """
    log_dir = base_dir or "/data/checkpoints/run_logs"
    os.makedirs(log_dir, exist_ok=True)

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    app_id = (os.environ.get("MODAL_APP_ID")
              or os.environ.get("MODAL_TASK_ID")
              or "noapp")
    tier_part = f"_tier{tier}" if tier is not None else ""
    fname     = f"{kind}{tier_part}_{ts}_{app_id}.log".replace("/", "_")
    log_path  = os.path.join(log_dir, fname)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
            return len(data)

        def flush(self):
            for s in self.streams:
                s.flush()

        def isatty(self):
            return False

    f = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    print(f"[logging] Writing combined stdout/stderr to: {log_path}")
    return log_path


def _prune_matching_files(
    dir_path: str,
    keep_basenames: set[str],
    matcher,
) -> int:
    """Delete files under dir_path that match matcher() unless explicitly kept.

    Returns the count of removed files.
    """
    removed = 0
    if not os.path.isdir(dir_path):
        return removed
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if not os.path.isfile(path) or not matcher(name) or name in keep_basenames:
            continue
        try:
            os.remove(path)
            removed += 1
        except Exception as e:
            print(f"[cleanup] Warning: could not remove {path}: {e}")
    return removed
