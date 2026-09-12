#!/usr/bin/env python
"""Resumable download of DESRES fast-folder reference trajectories.

The torchmd dataset README advertises ``http://`` URLs which return 403;
``https://`` serves the same objects and supports range requests, so every
transfer here resumes from whatever bytes already landed on disk.  That
matters because the host is subject to scheduled power cuts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://pub.htmd.org/protein_thermodynamics_data/reference_trajectories/"
CHUNK = 8 * 1024 * 1024


def remote_size(url: str) -> int:
    request = urllib.request.Request(url)
    request.add_header("Range", "bytes=0-0")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Range"].split("/")[-1])


def fetch(name: str, destination: Path, retries: int = 100) -> dict:
    url = f"{BASE}{name}_trajectories.tar.gz"
    target = destination / f"{name}_trajectories.tar.gz"
    total = remote_size(url)
    attempt = 0
    while True:
        have = target.stat().st_size if target.exists() else 0
        if have >= total:
            break
        attempt += 1
        if attempt > retries:
            raise RuntimeError(f"{name}: gave up at {have}/{total} bytes")
        request = urllib.request.Request(url)
        request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with target.open("ab") as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        handle.flush()
        except Exception as error:  # transient network faults resume next loop
            print(f"[{name}] retry {attempt}: {error!r}", flush=True)
            time.sleep(min(30, 2 * attempt))
        print(
            f"[{name}] {target.stat().st_size / 1e9:.2f} / {total / 1e9:.2f} GB",
            flush=True,
        )
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    record = {
        "name": name,
        "url": url,
        "path": str(target),
        "bytes": total,
        "sha256": digest.hexdigest(),
    }
    (destination / f"{name}.download.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{name}] complete {total / 1e9:.2f} GB sha256={record['sha256'][:16]}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--names", nargs="+", default=["bba", "homeodomain", "proteinb"])
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    records = [fetch(name, args.destination) for name in args.names]
    (args.destination / "download_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
