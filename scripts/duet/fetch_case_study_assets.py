#!/usr/bin/env python3
"""Fetch and selectively unpack public case-study assets.

The Zenodo bromodomain record stores all systems in two large tarballs.  This
utility supports parallel byte-range downloads and safe, pattern-filtered
extraction so only the files needed by the selected case are retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


CHUNK_BYTES = 4 * 1024 * 1024


def _request_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        return int(response.headers["Content-Length"])


def _download_range(url: str, start: int, end: int, destination: Path) -> int:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    written = 0
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        if response.status != 206:
            raise RuntimeError(f"Server ignored byte range {start}-{end}: HTTP {response.status}")
        while True:
            block = response.read(CHUNK_BYTES)
            if not block:
                break
            handle.write(block)
            written += len(block)
    expected = end - start + 1
    if written != expected:
        raise RuntimeError(f"Partial download {destination}: {written} != {expected}")
    return written


def parallel_download(url: str, destination: Path, workers: int, expected_size: int | None) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = expected_size or _request_size(url)
    if destination.exists() and destination.stat().st_size == size:
        print(f"already complete: {destination} ({size} bytes)")
        return

    with tempfile.TemporaryDirectory(prefix="duet-download-", dir=destination.parent) as temporary:
        temporary_path = Path(temporary)
        ranges = []
        width = (size + workers - 1) // workers
        for index in range(workers):
            start = index * width
            if start >= size:
                break
            end = min(size - 1, start + width - 1)
            ranges.append((index, start, end, temporary_path / f"part-{index:03d}"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = {
                executor.submit(_download_range, url, start, end, path): (index, start, end, path)
                for index, start, end, path in ranges
            }
            for future in concurrent.futures.as_completed(futures):
                index, start, end, _ = futures[future]
                written = future.result()
                print(f"part {index + 1}/{len(ranges)} complete: {start}-{end} ({written} bytes)")

        staging = destination.with_suffix(destination.suffix + ".partial")
        with staging.open("wb") as output:
            for _, _, _, path in ranges:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output, length=CHUNK_BYTES)
        if staging.stat().st_size != size:
            raise RuntimeError(f"Assembled download has {staging.stat().st_size} bytes; expected {size}")
        os.replace(staging, destination)
        print(f"downloaded: {destination} ({size} bytes)")


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            hasher.update(block)
    return hasher.hexdigest()


def verify(path: Path, checksum: str) -> None:
    algorithm, expected = checksum.split(":", 1)
    observed = digest(path, algorithm)
    if observed.lower() != expected.lower():
        raise RuntimeError(f"Checksum mismatch for {path}: {observed} != {expected}")
    print(f"verified {algorithm}: {observed}")


def _matches(name: str, patterns: list[str]) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in patterns)


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise RuntimeError(f"Unsafe archive member: {member_name}")
    return destination


def list_archive(path: Path, patterns: list[str]) -> list[str]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path, "r:*") as archive:
            names = archive.getnames()
    selected = [name for name in names if not patterns or _matches(name, patterns)]
    for name in selected:
        print(name)
    return selected


def extract_archive(path: Path, destination: Path, patterns: list[str]) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if patterns and not _matches(member.filename, patterns):
                    continue
                _safe_destination(destination, member.filename)
                archive.extract(member, destination)
                selected.append(member.filename)
    else:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if member.issym() or member.islnk():
                    continue
                if patterns and not _matches(member.name, patterns):
                    continue
                _safe_destination(destination, member.name)
                archive.extract(member, destination)
                selected.append(member.name)
    manifest = destination / "extracted_files.json"
    manifest.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"extracted {len(selected)} members to {destination}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("url")
    download.add_argument("destination", type=Path)
    download.add_argument("--workers", type=int, default=8)
    download.add_argument("--size", type=int)
    download.add_argument("--checksum")

    listing = subparsers.add_parser("list")
    listing.add_argument("archive", type=Path)
    listing.add_argument("--match", action="append", default=[])

    extract = subparsers.add_parser("extract")
    extract.add_argument("archive", type=Path)
    extract.add_argument("destination", type=Path)
    extract.add_argument("--match", action="append", default=[])

    arguments = parser.parse_args()
    if arguments.command == "download":
        if arguments.workers < 1 or arguments.workers > 32:
            parser.error("--workers must be between 1 and 32")
        parallel_download(arguments.url, arguments.destination, arguments.workers, arguments.size)
        if arguments.checksum:
            verify(arguments.destination, arguments.checksum)
    elif arguments.command == "list":
        list_archive(arguments.archive, arguments.match)
    else:
        extract_archive(arguments.archive, arguments.destination, arguments.match)


if __name__ == "__main__":
    main()
