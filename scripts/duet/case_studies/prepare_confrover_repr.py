#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def expand(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    from confrover.data.pretrain_repr import OpenFoldReprLoader

    cache_root = expand(config["model"]["cache_dir"])
    asset_root = expand("${DUET_ASSET_ROOT}/data/confrover_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    loader = OpenFoldReprLoader(repr_root=cache_root / "folding_repr")
    loader.generate_repr(
        seqres_index_pairs=[
            (str(config["trajectory"]["seqres"]), str(config["trajectory"]["case_id"]))
        ],
        msa_root=cache_root / "msa",
        openfold_params=asset_root / "openfold_params",
        save_struct=True,
        num_gpus=1,
        overwrite=False,
        msa_max_query_size=12,
    )
    loader.load(str(config["trajectory"]["seqres"]))
    print(f"representation_ready={cache_root / 'folding_repr'}")


if __name__ == "__main__":
    main()
