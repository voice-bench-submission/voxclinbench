"""`voxbench` command-line entrypoint.

Subcommands:
    voxbench fetch <dataset>
    voxbench eval --task <task_id> --predictions <path>
    voxbench eval --all --predictions-dir <dir> --out <path>
    voxbench compare --a <path> --b <path> --test delong

This is a thin dispatcher that delegates to voxbench.fetch and
voxbench.eval. Split-manifest binding (TODO: voxbench/splits/*.json) is
needed before `eval --all` can materialise a full leaderboard row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voxbench.eval import (
    delong_p,
    evaluate_task,
    holm_bonferroni,
    load_submission,
    paired_bootstrap_p,
)
from voxbench.fetch import fetch
from voxbench.tasks import TASKS, list_task_ids


def _cmd_fetch(args: argparse.Namespace) -> int:
    fetch(args.dataset, target=args.target)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    if args.all:
        if not args.predictions_dir:
            print("--all requires --predictions-dir", file=sys.stderr)
            return 2
        preds_dir = Path(args.predictions_dir)
        results = []
        for tid in list_task_ids():
            path = preds_dir / f"{tid}.json"
            if not path.exists():
                print(f"[skip] {tid}: no submission at {path}")
                continue
            submission = load_submission(path)
            res = evaluate_task(tid, submission["subject_probs"])
            results.append({
                "task_id": res.task_id,
                "auroc": res.auroc,
                "auprc": res.auprc,
                "ci95": list(res.ci95),
                "n_test": res.n_test,
            })
        out = Path(args.out) if args.out else Path("leaderboard_row.json")
        out.write_text(json.dumps(results, indent=2))
        print(f"wrote {out}")
        return 0

    if not args.task or not args.predictions:
        print("single-task mode requires --task and --predictions",
              file=sys.stderr)
        return 2
    submission = load_submission(args.predictions)
    res = evaluate_task(args.task, submission["subject_probs"])
    print(json.dumps({
        "task_id": res.task_id,
        "auroc": res.auroc,
        "auprc": res.auprc,
        "ci95": list(res.ci95),
        "n_test": res.n_test,
    }, indent=2))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    import numpy as np
    a = load_submission(args.a)
    b = load_submission(args.b)
    if a["task_id"] != b["task_id"]:
        print("submissions target different tasks", file=sys.stderr)
        return 2
    y_true = np.asarray([s["y_true"] for s in a["subject_probs"]], dtype=int)
    probs_a = np.asarray([s["y_prob"] for s in a["subject_probs"]], dtype=float)
    probs_b = np.asarray([s["y_prob"] for s in b["subject_probs"]], dtype=float)
    if args.test == "delong":
        auc_diff, p = delong_p(y_true, probs_a, probs_b)
    elif args.test == "paired-bootstrap":
        auc_diff, p = paired_bootstrap_p(y_true, probs_a, probs_b)
    else:
        print(f"unknown test {args.test!r}", file=sys.stderr)
        return 2
    print(json.dumps({
        "task_id": a["task_id"],
        "test": args.test,
        "auc_diff": auc_diff,
        "p": p,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voxbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("dataset")
    p_fetch.add_argument("--target", default=None)
    p_fetch.set_defaults(func=_cmd_fetch)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--task", default=None)
    p_eval.add_argument("--predictions", default=None)
    p_eval.add_argument("--all", action="store_true")
    p_eval.add_argument("--predictions-dir", default=None)
    p_eval.add_argument("--out", default=None)
    p_eval.set_defaults(func=_cmd_eval)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--a", required=True)
    p_cmp.add_argument("--b", required=True)
    p_cmp.add_argument("--test", default="delong",
                       choices=["delong", "paired-bootstrap"])
    p_cmp.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
