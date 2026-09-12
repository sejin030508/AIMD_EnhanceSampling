#!/usr/bin/env python
"""Evaluate timing-variation runs whose sampling finished but evaluation did not.

The cell runner builds its evaluation path from a stage string it computes
itself (``stride<N>_t32``), while the sampler writes to the stage named in the
config (``stride<N>_t32_<tag>``).  With per-timing configs those disagree, so
evaluation was handed a directory that does not exist and failed after the
sampling had already succeeded.  The generated populations are intact, so this
re-runs only the evaluation, against the directory the sampler actually used.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()

    pending = []
    for metrics in sorted(args.root.rglob("metrics.json")):
        run_dir = metrics.parent
        if (run_dir / "small_protein_metrics.json").exists():
            continue
        # .../<tag>/<molecule>/<stage>/<method>/seed_<seed>/metrics.json
        molecule = run_dir.parents[2].name
        pending.append((run_dir, molecule))

    print(f"pending evaluations: {len(pending)}", flush=True)
    results = []
    for run_dir, molecule in pending:
        prepared = args.prepared_root / molecule
        started = time.time()
        process = subprocess.run(
            [
                args.python,
                str(args.code_root / "evaluate_small_protein_run.py"),
                "--run-dir", str(run_dir),
                "--prepared-dir", str(prepared),
            ],
            capture_output=True,
            text=True,
        )
        elapsed = round(time.time() - started, 1)
        ok = process.returncode == 0
        label = str(run_dir).replace(str(args.root) + "/", "")
        print(f"{'OK ' if ok else 'FAIL'} {elapsed:7.1f}s  {label}", flush=True)
        if not ok:
            tail = (process.stderr or process.stdout).strip().splitlines()[-6:]
            for line in tail:
                print(f"        {line}", flush=True)
        results.append({"run_dir": str(run_dir), "ok": ok, "seconds": elapsed})

    summary = args.root / "evaluation_recovery.json"
    summary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    failures = [item for item in results if not item["ok"]]
    print(f"done: {len(results) - len(failures)} ok, {len(failures)} failed", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
