#!/usr/bin/env python
"""Preflight: can this environment load the published PVB checkpoint?

The checkpoint is a pickled model object, not a state dict, so loading it
exercises the whole import chain and the torch version compatibility at once.
Failures are reported with the specific missing module rather than a traceback
so the environment can be fixed in one pass.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    try:
        import torch
    except Exception as error:
        print("FAIL torch:", error)
        return 1
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

    try:
        model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        print(f"FAIL missing module: {error.name}")
        return 2
    except Exception:
        traceback.print_exc(limit=6)
        return 3

    print("LOADED", type(model).__module__ + "." + type(model).__name__)
    for attribute in (
        "sigma", "using_ode", "backbone", "hidden_dim", "layers",
        "cutoff_lower", "cutoff_upper", "cutoff_H", "k_neighbors", "model_type",
    ):
        if hasattr(model, attribute):
            print(f"  {attribute:14s} {getattr(model, attribute)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  parameters     {total:,}")
    print("  has inference():", hasattr(model, "inference"))
    print("  has encode()/decode():", hasattr(model, "encode"), hasattr(model, "decode"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
