from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from confmh.config import load_config, resolve_path
from confmh.discovery.structure import (
    aligned_local_rmsd,
    kabsch_transform,
    read_pdb,
    sequence,
    sequence_alignment_pairs,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _in_ranges(resseq: int, ranges: list[list[int | None]] | None) -> bool:
    if not ranges:
        return True
    return any(
        (-10**9 if begin is None else int(begin)) <= resseq <= (10**9 if end is None else int(end))
        for begin, end in ranges
    )


def _mapped_ca(sample_pdb: Path, start_pdb: Path, target_pdb: Path):
    sample, start, target = read_pdb(sample_pdb), read_pdb(start_pdb), read_pdb(target_pdb)
    sample_start = sequence_alignment_pairs(sequence(sample), sequence(start))
    start_target = dict(sequence_alignment_pairs(sequence(start), sequence(target)))
    triples = [
        (sample_i, start_i, start_target[start_i])
        for sample_i, start_i in sample_start
        if start_i in start_target
        and "CA" in sample[sample_i].atoms
        and "CA" in start[start_i].atoms
        and "CA" in target[start_target[start_i]].atoms
    ]
    if len(triples) < 3:
        raise ValueError(f"Only {len(triples)} common C-alpha atoms across sample/start/target")
    return sample, start, target, triples


def alternate_state_metrics(
    sample_pdb: str | Path,
    start_pdb: str | Path,
    target_pdb: str | Path,
    *,
    local_residinfo: str | Path | None = None,
) -> dict[str, float | int]:
    sample_pdb, start_pdb, target_pdb = map(lambda p: Path(p).expanduser().resolve(), (sample_pdb, start_pdb, target_pdb))
    sample, start, target, triples = _mapped_ca(sample_pdb, start_pdb, target_pdb)
    alignment_ranges = metric_ranges = None
    if local_residinfo is not None:
        local = json.loads(Path(local_residinfo).read_text(encoding="utf-8"))
        alignment_ranges = local.get("alignment_resid_ranges")
        metric_ranges = local.get("metric_resid_ranges")
    alignment = [triple for triple in triples if _in_ranges(target[triple[2]].resseq, alignment_ranges)]
    metric = [triple for triple in triples if _in_ranges(target[triple[2]].resseq, metric_ranges)]
    if len(alignment) < 3 or len(metric) < 3:
        raise ValueError(f"Local selection too small: alignment={len(alignment)}, metric={len(metric)}")

    def coords(residues, triples, position):
        return np.stack([residues[triple[position]].atoms["CA"] for triple in triples])

    sample_align, target_align = coords(sample, alignment, 0), coords(target, alignment, 2)
    sample_metric, target_metric = coords(sample, metric, 0), coords(target, metric, 2)
    start_align, start_metric = coords(start, alignment, 1), coords(start, metric, 1)
    target_rmsd = aligned_local_rmsd(sample_align, target_align, sample_metric, target_metric)
    start_rmsd = aligned_local_rmsd(sample_align, start_align, sample_metric, start_metric)
    endpoint_distance = aligned_local_rmsd(start_align, target_align, start_metric, target_metric)
    rotation, translation = kabsch_transform(sample_align, target_align)
    distances = np.linalg.norm(sample_metric @ rotation + translation - target_metric, axis=1)
    length = len(metric)
    d0 = max(0.5, 1.24 * np.cbrt(max(length - 15, 1)) - 1.8)
    tm_like = float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))
    return {
        "rmsd_start_a": start_rmsd,
        "rmsd_target_a": target_rmsd,
        "endpoint_distance_a": endpoint_distance,
        "half_endpoint_threshold_a": 0.5 * endpoint_distance,
        "ca_tm_score": tm_like,
        "n_alignment_residues": len(alignment),
        "n_metric_residues": len(metric),
    }


def analyze_alternate_run(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    cfg = load_config(run_dir / "resolved_config.yaml")
    rows = _load_jsonl(run_dir / "candidates.jsonl")
    start_pdb = resolve_path(
        cfg, cfg["benchmark"].get("evaluation_start_pdb", cfg["system"]["start_pdb"])
    )
    target_pdb = resolve_path(cfg, cfg["benchmark"]["target_pdb"])
    local_info = cfg["benchmark"].get("local_residinfo")
    local_info_path = None if local_info is None else resolve_path(cfg, local_info)
    analyzed: list[dict[str, Any]] = []
    for row in rows:
        metrics = alternate_state_metrics(
            Path(row["candidate_pdb"]), start_pdb, target_pdb, local_residinfo=local_info_path
        )
        result = {**row, **metrics}
        result["hit_3a"] = metrics["rmsd_target_a"] <= 3.0
        result["hit_half_endpoint"] = metrics["rmsd_target_a"] <= metrics["half_endpoint_threshold_a"]
        result["valid_hit_3a"] = bool(result["strict_valid"] and result["hit_3a"])
        result["valid_hit_half_endpoint"] = bool(
            result["strict_valid"] and result["hit_half_endpoint"]
        )
        analyzed.append(result)
    fieldnames = sorted({key for row in analyzed for key in row if key != "validity"})
    csv_path = run_dir / "alternate_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in analyzed])

    def first_call(key: str) -> int | None:
        calls = [int(row["model_call"]) for row in analyzed if row[key]]
        return min(calls) if calls else None

    walltime_by_batch: dict[int, float] = {}
    for row in analyzed:
        walltime_by_batch[int(row["batch_step"])] = float(row["proposal_batch_walltime_s"])
    gpu_hours = sum(walltime_by_batch.values()) / 3600.0
    n = len(analyzed)
    first_3a, first_half = first_call("valid_hit_3a"), first_call("valid_hit_half_endpoint")
    summary = {
        "model_backend": cfg["model"].get("backend", "confrover"),
        "condition": run_dir.parent.parent.name,
        "seed_id": run_dir.name,
        "random_seed": int(cfg["run"]["seed"]),
        "protocol": cfg["protocol"]["name"],
        "controller": cfg["controller"]["kind"],
        "category": cfg["benchmark"]["category"],
        "test_case": cfg["benchmark"]["test_case"],
        "start_id": cfg["benchmark"]["start_id"],
        "target_id": cfg["benchmark"]["target_id"],
        "benchmark_id": (
            f'{cfg["benchmark"]["category"]}:'
            f'{cfg["benchmark"]["start_id"]}->{cfg["benchmark"]["target_id"]}'
        ),
        "model_calls": n,
        "endpoint_distance_a": analyzed[0]["endpoint_distance_a"] if analyzed else None,
        "alternate_hit_rate_3a": float(np.mean([row["hit_3a"] for row in analyzed])) if analyzed else 0.0,
        "alternate_hit_rate_half_endpoint": float(np.mean([row["hit_half_endpoint"] for row in analyzed])) if analyzed else 0.0,
        "valid_hit_rate_3a": float(np.mean([row["valid_hit_3a"] for row in analyzed])) if analyzed else 0.0,
        "valid_hit_rate_half_endpoint": float(np.mean([row["valid_hit_half_endpoint"] for row in analyzed])) if analyzed else 0.0,
        "median_target_rmsd_a": float(np.median([row["rmsd_target_a"] for row in analyzed])) if analyzed else None,
        "best_valid_target_rmsd_a": min(
            (float(row["rmsd_target_a"]) for row in analyzed if row["strict_valid"]), default=None
        ),
        "best_ca_tm_score": max((float(row["ca_tm_score"]) for row in analyzed), default=None),
        "proposals_to_first_valid_hit_3a": first_3a,
        "proposals_to_first_valid_hit_half_endpoint": first_half,
        "coverage_auc_3a": 0.0 if first_3a is None else (n - first_3a + 1) / n,
        "coverage_auc_half_endpoint": 0.0 if first_half is None else (n - first_half + 1) / n,
        "strict_valid_rate": float(np.mean([row["strict_valid"] for row in analyzed])) if analyzed else 0.0,
        "paper_compliant_rate_0_90": float(
            np.mean([row.get("paper_compliant_0_90", row["strict_valid"]) for row in analyzed])
        ) if analyzed else 0.0,
        "strengthened_valid_rate": float(
            np.mean([row.get("strengthened_valid", False) for row in analyzed])
        ) if analyzed else 0.0,
        "safety_valid_rate": float(np.mean([row["safety_valid"] for row in analyzed])) if analyzed else 0.0,
        "measured_gpu_hours": gpu_hours,
        "valid_unique_hits_3a_per_gpu_hour": (
            sum(row["valid_hit_3a"] for row in analyzed) / gpu_hours if gpu_hours > 0 else None
        ),
        "mh_used": False,
        "target_was_hidden_during_generation": cfg["protocol"]["name"] == "target_blind",
    }
    (run_dir / "alternate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir / "alternate_summary.json"


def aggregate_alternate_runs(run_dirs: Iterable[str | Path], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for directory in run_dirs:
        directory = Path(directory).expanduser().resolve()
        path = directory / "alternate_summary.json"
        if not path.exists():
            path = analyze_alternate_run(path.parent)
        summary = json.loads(path.read_text(encoding="utf-8"))
        # Older per-run summaries predate explicit condition and pair identifiers.
        # Recover both from the run layout/metadata so distinct CV ablations and
        # structure pairs sharing a UniProt accession are never conflated.
        summary.setdefault("condition", directory.parent.parent.name)
        summary.setdefault("seed_id", directory.name)
        summary.setdefault(
            "benchmark_id",
            f'{summary["category"]}:{summary["start_id"]}->{summary["target_id"]}',
        )
        summaries.append(summary)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for summary in summaries:
        key = (
            summary["model_backend"],
            summary["condition"],
            summary["protocol"],
            summary["controller"],
        )
        groups.setdefault(key, []).append(summary)
    aggregate = []
    for (model, condition, protocol, controller), rows in sorted(groups.items()):
        aggregate.append(
            {
                "model_backend": model,
                "condition": condition,
                "protocol": protocol,
                "controller": controller,
                "n_runs": len(rows),
                "n_test_cases": len({row["benchmark_id"] for row in rows}),
                "case_seed_success_rate_3a": float(np.mean([row["valid_hit_rate_3a"] > 0 for row in rows])),
                "case_seed_success_rate_half_endpoint": float(
                    np.mean([row["valid_hit_rate_half_endpoint"] > 0 for row in rows])
                ),
                "median_best_valid_target_rmsd_a": float(
                    np.median(
                        [row["best_valid_target_rmsd_a"] for row in rows if row["best_valid_target_rmsd_a"] is not None]
                    )
                ) if any(row["best_valid_target_rmsd_a"] is not None for row in rows) else None,
                "mean_strict_valid_rate": float(np.mean([row["strict_valid_rate"] for row in rows])),
                "total_gpu_hours": float(sum(row["measured_gpu_hours"] for row in rows)),
            }
        )
    payload = {
        "runs": summaries,
        "aggregate": aggregate,
        "reporting_rule": "All preregistered compatible cases and seeds are included; examples are secondary.",
    }
    path = output_dir / "alternate_benchmark_summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
