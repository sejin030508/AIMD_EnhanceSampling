#!/usr/bin/env python
"""Report the hyper-parameters of a shipped TPS-DPS ``tica_model.pkl``.

Refitting TICA for a new protein only produces comparable numbers if the
lag time, output dimension and scaling match the models the benchmark
ships.  The official repository publishes the fitted objects but not the
fitting script, so read the settings back off the pickles instead of
guessing them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def describe(path: Path) -> dict:
    model = joblib.load(path)
    record: dict = {"path": str(path), "type": type(model).__name__}
    for attribute in (
        "lag",
        "dim",
        "var_cutoff",
        "kinetic_map",
        "commute_map",
        "epsilon",
        "reversible",
        "scaling",
        "ndim",
        "data_producer",
    ):
        if not hasattr(model, attribute):
            continue
        try:
            value = getattr(model, attribute)
        except Exception as error:  # some properties need a fitted producer
            record[attribute] = f"<unavailable: {error!r}>"
            continue
        if isinstance(value, (int, float, bool, str)) or value is None:
            record[attribute] = value
        else:
            record[attribute] = f"<{type(value).__name__}>"
    for attribute in ("eigenvalues", "timescales", "cumvar", "mean"):
        try:
            value = np.asarray(getattr(model, attribute))
        except Exception:
            continue
        record[attribute] = {
            "shape": list(value.shape),
            "head": np.round(value.ravel()[:8], 6).tolist(),
        }
    try:
        record["input_dimension"] = int(np.asarray(model.mean).shape[0])
    except Exception:
        pass
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    for path in args.models:
        try:
            records.append(describe(path))
        except Exception as error:
            records.append({"path": str(path), "error": repr(error)})
    text = json.dumps(records, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
