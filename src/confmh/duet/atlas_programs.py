from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from confmh.pca_cv import PCACV, fit_reference_pca


def _load_design_trajectories(
    topology: Path,
    trajectories: Sequence[Path],
    *,
    stride: int,
    max_frames_per_trajectory: int | None,
):
    import mdtraj as md

    topology_frame = md.load(str(topology))
    protein = topology_frame.topology.select("protein and chainid 0")
    reference = topology_frame.atom_slice(protein)
    ca = reference.topology.select("name CA")
    loaded = []
    for path in trajectories:
        traj = md.load(str(path), top=str(topology), atom_indices=protein, stride=stride)
        if max_frames_per_trajectory is not None:
            traj = traj[: int(max_frames_per_trajectory)]
        traj.superpose(reference, atom_indices=ca, ref_atom_indices=ca)
        loaded.append(traj)
    return reference, ca, loaded


def _transition_segments(
    scores: Sequence[np.ndarray],
    start_interval: tuple[float, float],
    target_interval: tuple[float, float],
    frames: int,
) -> list[np.ndarray]:
    segments = []
    for values in scores:
        starts = np.flatnonzero((values >= start_interval[0]) & (values <= start_interval[1]))
        for start in starts:
            targets = np.flatnonzero(
                (np.arange(len(values)) > start)
                & (values >= target_interval[0])
                & (values <= target_interval[1])
            )
            if len(targets) == 0:
                continue
            end = int(targets[0])
            if end - int(start) < frames - 1:
                continue
            segments.append(np.rint(np.linspace(start, end, frames)).astype(int))
            break
    return segments


def _select_ordered_contacts(
    trajectories,
    ca_indices: np.ndarray,
    segments: Sequence[np.ndarray],
    *,
    min_sequence_separation: int,
    min_change_nm: float,
    min_event_separation: int,
    stable_fraction: float,
    max_event_frame: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not segments:
        raise RuntimeError("No R1/R2 start-to-target transition segments were found")
    candidates = []
    pairs = itertools.combinations(range(len(ca_indices)), 2)
    for i, j in pairs:
        if j - i < min_sequence_separation:
            continue
        traces = []
        for trajectory, indices in zip(trajectories, segments):
            coords = trajectory.xyz[indices][:, ca_indices, :]
            traces.append(np.linalg.norm(coords[:, i] - coords[:, j], axis=-1))
        if not traces:
            continue
        values = np.stack(traces)
        start_mean, end_mean = float(values[:, 0].mean()), float(values[:, -1].mean())
        change = end_mean - start_mean
        if abs(change) < min_change_nm:
            continue
        threshold = 0.5 * (start_mean + end_mean)
        crossing = []
        for trace in values:
            hits = np.flatnonzero(trace <= threshold) if change < 0 else np.flatnonzero(trace >= threshold)
            if len(hits):
                crossing.append(int(hits[0]))
        stability = len(crossing) / len(values)
        if stability < stable_fraction:
            continue
        median_crossing = float(np.median(crossing))
        # Ordered events must be completed before the terminal-only frame.  A
        # crossing at T cannot be made valid by silently clipping its window to
        # T-1, because that would change the reference-derived event.
        if median_crossing > max_event_frame:
            continue
        candidates.append(
            {
                "residue_i": i,
                "residue_j": j,
                "threshold_nm": threshold,
                "target_contact": 1 if change < 0 else 0,
                "change_nm": change,
                "stability": stability,
                "median_crossing": median_crossing,
                "crossings": crossing,
            }
        )
    candidates.sort(key=lambda row: (row["stability"], abs(row["change_nm"])), reverse=True)
    for left in candidates:
        for right in candidates:
            if right["median_crossing"] - left["median_crossing"] < min_event_separation:
                continue
            if {left["residue_i"], left["residue_j"]} == {right["residue_i"], right["residue_j"]}:
                continue
            return [left, right], {"candidate_count": len(candidates), "top_candidates": candidates[:50]}
    raise RuntimeError(
        f"No two stable contact events separated by {min_event_separation} physical frames; "
        f"retained {len(candidates)} candidates"
    )


def prepare_atlas_programs(cfg: dict[str, Any]) -> Path:
    reference = cfg["reference"]
    output = cfg["output"]
    topology = Path(reference["topology"]).expanduser().resolve()
    trajectories = [Path(path).expanduser().resolve() for path in reference["design_trajectories"]]
    output_dir = Path(output["directory"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    design_pca_dir = output_dir / "design_pca_r1_r2"
    model_path = design_pca_dir / "pca_cv.npz"
    if not model_path.exists():
        fit_reference_pca(
            topology_pdb=topology,
            trajectories=trajectories,
            output_dir=design_pca_dir,
            protein_selection=str(reference.get("protein_selection", "protein and chainid 0")),
            stride=int(reference.get("stride", 10)),
            max_frames_per_trajectory=reference.get("max_frames_per_trajectory"),
            n_components=int(reference.get("n_components", 3)),
            n_seed_frames=0,
        )
    pca = PCACV.load(model_path)
    _, ca_indices, loaded = _load_design_trajectories(
        topology,
        trajectories,
        stride=int(reference.get("stride", 10)),
        max_frames_per_trajectory=reference.get("max_frames_per_trajectory"),
    )
    scores = [pca.project_trajectory(trajectory)[:, 0] for trajectory in loaded]
    all_scores = np.concatenate(scores)
    start_pdb = Path(reference["initial_structure"]).expanduser().resolve()
    start_score = float(pca.project_files(start_pdb)[0, 0])
    width = float(reference.get("basin_half_width", 0.25))
    start_interval = (start_score - width, start_score + width)
    lower, upper = np.quantile(all_scores, [0.10, 0.90])
    target_center = float(upper if abs(upper - start_score) >= abs(lower - start_score) else lower)
    target_interval = (target_center - width, target_center + width)
    middle = 0.5 * (start_score + target_center)
    intermediate_interval = (middle - width, middle + width)
    horizon = int(cfg.get("trajectory_horizon", 8))
    segments_by_trajectory = []
    segment_trajectories = []
    for trajectory, values in zip(loaded, scores):
        segments = _transition_segments([values], start_interval, target_interval, horizon + 1)
        if segments:
            segments_by_trajectory.append(segments[0])
            segment_trajectories.append(trajectory)
    selected, report = _select_ordered_contacts(
        segment_trajectories,
        ca_indices,
        segments_by_trajectory,
        min_sequence_separation=int(cfg.get("contact_selection", {}).get("min_sequence_separation", 5)),
        min_change_nm=float(cfg.get("contact_selection", {}).get("min_change_nm", 0.15)),
        min_event_separation=int(cfg.get("contact_selection", {}).get("min_event_separation", 2)),
        stable_fraction=float(cfg.get("contact_selection", {}).get("stable_fraction", 0.5)),
        max_event_frame=horizon - 1,
    )
    observables: dict[str, Any] = {
        "pc1": {"kind": "pc1", "pca_model": str(model_path), "component": 0}
    }
    events = []
    for name, row in zip(("contact_A", "contact_B"), selected):
        observables[name] = {
            "kind": "contact",
            "residue_i": row["residue_i"],
            "residue_j": row["residue_j"],
            "threshold_nm": row["threshold_nm"],
        }
        center = int(round(row["median_crossing"]))
        events.append(
            {
                "name": name,
                "observable": name,
                "target_interval": [row["target_contact"], row["target_contact"]],
                "physical_window": [max(1, center - 1), min(horizon - 1, center + 1)],
            }
        )
    terminal = {
        "name": "terminal_basin",
        "observable": "pc1",
        "target_interval": list(target_interval),
        "physical_window": [horizon, horizon],
    }
    catalog = {
        "design_split": {"task_design": ["R1", "R2"], "held_out_evaluation": ["R3"]},
        "observables": observables,
        "tasks": {
            "endpoint": {"type": "terminal", "events": [terminal]},
            "windowed": {
                "type": "windowed",
                "events": [
                    {
                        "name": "intermediate_pc1",
                        "observable": "pc1",
                        "target_interval": list(intermediate_interval),
                        "physical_window": [3, 5],
                    }
                ],
                "terminal_event": terminal,
            },
            "ordered": {
                "type": "ordered",
                "allow_same_frame": False,
                "events": events,
                "terminal_event": terminal,
            },
        },
        "selection": {
            "start_pc1": start_score,
            "start_interval": list(start_interval),
            "target_interval": list(target_interval),
            "intermediate_interval": list(intermediate_interval),
            "transition_segment_count": len(segments_by_trajectory),
            "selected_contacts": selected,
            **report,
        },
    }
    catalog_path = output_dir / "duet_programs.yaml"
    with catalog_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(catalog, handle, sort_keys=False)
    with (output_dir / "selection_report.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(catalog["selection"], handle, sort_keys=False)
    return catalog_path
