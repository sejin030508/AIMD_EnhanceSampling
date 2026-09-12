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


def _load_initial_ca(path: Path) -> np.ndarray:
    import mdtraj as md

    trajectory = md.load(str(path))
    protein = trajectory.topology.select("protein and chainid 0")
    trajectory = trajectory.atom_slice(protein)
    ca = trajectory.topology.select("name CA")
    return np.asarray(trajectory.xyz[0, ca], dtype=float)


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


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/end indices for contiguous true runs."""
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _nearest_time_indices(times_ps: np.ndarray, targets_ps: np.ndarray) -> np.ndarray:
    """Map target times to nearest saved trajectory frames."""
    times = np.asarray(times_ps, dtype=float)
    targets = np.asarray(targets_ps, dtype=float)
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    choose_left = np.abs(times[left] - targets) <= np.abs(times[right] - targets)
    return np.where(choose_left, left, right).astype(int)


def _transition_path_segment(
    values: np.ndarray,
    times_ps: np.ndarray,
    start_interval: tuple[float, float],
    target_interval: tuple[float, float],
    *,
    frames: int,
    model_step_ps: float,
    start_dwell_steps: float,
    target_dwell_steps: float,
    duration_ratio: tuple[float, float],
    max_time_error_steps: float,
    start_match_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Extract a last-start-exit to stable-target-entry path on a fixed time grid.

    Unlike ``_transition_segments``, this does not stretch an arbitrary interval
    to the model horizon.  It samples existing frames nearest to the physical
    times implied by the model lag.
    """
    scores = np.asarray(values, dtype=float)
    times = np.asarray(times_ps, dtype=float)
    if scores.ndim != 1 or times.ndim != 1 or len(scores) != len(times):
        raise ValueError("scores and times_ps must be one-dimensional and equal length")
    if len(scores) < 2 or frames < 2 or model_step_ps <= 0.0:
        return None
    if start_dwell_steps <= 0.0 or target_dwell_steps <= 0.0:
        raise ValueError("start and target dwell times must be positive")
    if max_time_error_steps < 0.0:
        raise ValueError("max_time_error_steps must be non-negative")
    deltas = np.diff(times)
    if np.any(deltas <= 0.0):
        raise ValueError("trajectory times must be strictly increasing")
    saved_step_ps = float(np.median(deltas))
    start_frames = max(
        2, int(np.ceil(start_dwell_steps * model_step_ps / saved_step_ps)) + 1
    )
    target_frames = max(
        2, int(np.ceil(target_dwell_steps * model_step_ps / saved_step_ps)) + 1
    )

    in_start = (scores >= start_interval[0]) & (scores <= start_interval[1])
    if start_match_mask is not None:
        match = np.asarray(start_match_mask, dtype=bool)
        if match.shape != in_start.shape:
            raise ValueError("start_match_mask must match the score trajectory")
        in_start &= match
    in_target = (scores >= target_interval[0]) & (scores <= target_interval[1])
    start_runs = [run for run in _true_runs(in_start) if run[1] - run[0] + 1 >= start_frames]
    target_runs = [run for run in _true_runs(in_target) if run[1] - run[0] + 1 >= target_frames]
    expected_duration_ps = float((frames - 1) * model_step_ps)

    for target_entry, target_end in target_runs:
        prior_start_runs = [run for run in start_runs if run[1] < target_entry]
        if not prior_start_runs:
            continue
        start_run = prior_start_runs[-1]
        start_exit = int(start_run[1])
        observed_duration_ps = float(times[target_entry] - times[start_exit])
        ratio = observed_duration_ps / expected_duration_ps
        if ratio < duration_ratio[0] or ratio > duration_ratio[1]:
            continue
        grid_ps = times[start_exit] + np.arange(frames, dtype=float) * model_step_ps
        if grid_ps[-1] > times[-1]:
            continue
        indices = _nearest_time_indices(times, grid_ps)
        if np.any(np.diff(indices) <= 0):
            continue
        errors = np.abs(times[indices] - grid_ps)
        max_time_error_ps = float(np.max(errors))
        if max_time_error_ps > max_time_error_steps * model_step_ps:
            continue
        # The model-time endpoint must still be in the target state.  A brief
        # target hit followed by exit is not a valid endpoint reference.
        if not bool(in_target[indices[-1]]):
            continue
        return indices, {
            "start_run": [int(start_run[0]), int(start_run[1])],
            "target_run": [int(target_entry), int(target_end)],
            "start_dwell_required_frames": start_frames,
            "target_dwell_required_frames": target_frames,
            "start_exit_index": start_exit,
            "target_entry_index": int(target_entry),
            "sampled_end_index": int(indices[-1]),
            "start_time_ps": float(times[start_exit]),
            "target_entry_time_ps": float(times[target_entry]),
            "sampled_end_time_ps": float(times[indices[-1]]),
            "transition_path_duration_ps": observed_duration_ps,
            "model_horizon_duration_ps": expected_duration_ps,
            "duration_ratio": float(ratio),
            "saved_frame_step_ps": saved_step_ps,
            "max_sampling_time_error_ps": max_time_error_ps,
            "sampled_indices": indices.tolist(),
            "sampled_times_ps": times[indices].tolist(),
        }
    return None


def _transition_path_geometry_segment(
    values: np.ndarray,
    times_ps: np.ndarray,
    start_interval: tuple[float, float],
    target_interval: tuple[float, float],
    *,
    analysis_frames: int,
    start_dwell_ps: float,
    target_dwell_ps: float,
    start_match_mask: np.ndarray | None = None,
    not_before_time_ps: float | None = None,
    target_not_after_time_ps: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Extract last-exit route geometry without matching generated timestamps.

    The returned route contains ``analysis_frames`` existing trajectory frames
    sampled across the observed last-start-exit to stable-target-entry segment.
    Coordinates are never interpolated.  The sampling grid is only a common
    route-analysis index used to select and compare A/B milestones.
    """
    scores = np.asarray(values, dtype=float)
    times = np.asarray(times_ps, dtype=float)
    if scores.ndim != 1 or times.ndim != 1 or len(scores) != len(times):
        raise ValueError("scores and times_ps must be one-dimensional and equal length")
    if len(scores) < 2 or int(analysis_frames) < 2:
        return None
    if float(start_dwell_ps) <= 0.0 or float(target_dwell_ps) <= 0.0:
        raise ValueError("start_dwell_ps and target_dwell_ps must be positive")
    deltas = np.diff(times)
    if np.any(deltas <= 0.0):
        raise ValueError("trajectory times must be strictly increasing")
    saved_step_ps = float(np.median(deltas))
    start_frames = max(2, int(np.ceil(float(start_dwell_ps) / saved_step_ps)) + 1)
    target_frames = max(2, int(np.ceil(float(target_dwell_ps) / saved_step_ps)) + 1)

    in_start = (scores >= start_interval[0]) & (scores <= start_interval[1])
    if start_match_mask is not None:
        match = np.asarray(start_match_mask, dtype=bool)
        if match.shape != in_start.shape:
            raise ValueError("start_match_mask must match the score trajectory")
        in_start &= match
    in_target = (scores >= target_interval[0]) & (scores <= target_interval[1])
    start_runs = [run for run in _true_runs(in_start) if run[1] - run[0] + 1 >= start_frames]
    target_runs = [run for run in _true_runs(in_target) if run[1] - run[0] + 1 >= target_frames]
    not_before = float(times[0]) if not_before_time_ps is None else float(not_before_time_ps)

    for target_entry, target_end in target_runs:
        if times[target_entry] <= not_before:
            continue
        if (
            target_not_after_time_ps is not None
            and times[target_entry] > float(target_not_after_time_ps)
        ):
            continue
        prior_start_runs = [
            run
            for run in start_runs
            if run[1] < target_entry and times[run[1]] >= not_before
        ]
        if not prior_start_runs:
            continue
        start_run = prior_start_runs[-1]
        start_exit = int(start_run[1])
        native_indices = np.arange(start_exit, int(target_entry) + 1, dtype=int)
        if len(native_indices) < int(analysis_frames):
            continue
        route_offsets = np.rint(
            np.linspace(0, len(native_indices) - 1, int(analysis_frames))
        ).astype(int)
        indices = native_indices[route_offsets]
        if np.any(np.diff(indices) <= 0):
            continue
        initial_index = int(np.argmin(np.abs(times - not_before)))
        if abs(float(times[initial_index]) - not_before) > 0.51 * saved_step_ps:
            continue
        return indices, {
            "start_run": [int(start_run[0]), int(start_run[1])],
            "target_run": [int(target_entry), int(target_end)],
            "start_dwell_required_frames": start_frames,
            "target_dwell_required_frames": target_frames,
            "initial_index": initial_index,
            "initial_time_ps": float(times[initial_index]),
            "start_exit_index": start_exit,
            "target_entry_index": int(target_entry),
            "start_exit_time_ps": float(times[start_exit]),
            "target_entry_time_ps": float(times[target_entry]),
            "reference_first_passage_ps": float(times[target_entry] - times[initial_index]),
            "reference_transition_path_ps": float(times[target_entry] - times[start_exit]),
            "saved_frame_step_ps": saved_step_ps,
            "native_path_frame_count": int(len(native_indices)),
            "analysis_frame_count": int(analysis_frames),
            "sampled_indices": indices.tolist(),
            "sampled_times_ps": times[indices].tolist(),
            "coordinate_interpolation": False,
        }
    return None


def _select_model_lag_in_10ps(
    reference_first_passage_ps: float,
    *,
    horizon: int,
    target_compression: float = 2.0,
    minimum: int = 20,
    maximum: int = 210,
) -> int:
    """Choose an in-training-range lag before any generated method outcome."""
    if reference_first_passage_ps <= 0.0 or horizon < 1 or target_compression <= 0.0:
        raise ValueError("first passage, horizon, and target compression must be positive")
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid lag bounds")
    raw = int(np.floor(reference_first_passage_ps / (target_compression * horizon * 10.0)))
    return int(np.clip(raw, minimum, maximum))


def _ca_rmsd_to_reference(ca_nm: np.ndarray, reference_ca_nm: np.ndarray) -> np.ndarray:
    """Kabsch-aligned C-alpha RMSD of each frame to one initial structure."""
    frames = np.asarray(ca_nm, dtype=float)
    reference = np.asarray(reference_ca_nm, dtype=float)
    if frames.ndim != 3 or reference.ndim != 2 or frames.shape[1:] != reference.shape:
        raise ValueError(
            "C-alpha RMSD expects frames (time, residues, 3) and a matching reference"
        )
    ref_centered = reference - reference.mean(axis=0, keepdims=True)
    values = []
    for frame in frames:
        centered = frame - frame.mean(axis=0, keepdims=True)
        u, _, vt = np.linalg.svd(centered.T @ ref_centered)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        aligned = centered @ rotation
        values.append(np.sqrt(np.mean(np.sum((aligned - ref_centered) ** 2, axis=-1))))
    return np.asarray(values, dtype=float)


def _first_persistent_crossing(mask: np.ndarray, consecutive: int) -> int | None:
    """Return the first frame completing a persistent threshold crossing."""
    required = max(1, int(consecutive))
    for start, end in _true_runs(np.asarray(mask, dtype=bool)):
        if end - start + 1 >= required:
            return int(start + required - 1)
    return None


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
    initial_ca_nm: np.ndarray | None = None,
    min_initial_margin_nm: float = 0.0,
    discovery_segment_index: int | None = None,
    min_event_support_count: int | None = None,
    min_pair_support_count: int | None = None,
    threshold_statistic: str = "mean",
    min_crossing_persistence: int = 1,
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
        if discovery_segment_index is not None:
            if not 0 <= int(discovery_segment_index) < len(values):
                raise ValueError("discovery_segment_index is outside the retained segments")
            definition_values = values[[int(discovery_segment_index)]]
        else:
            definition_values = values
        if threshold_statistic == "median":
            start_center = float(np.median(definition_values[:, 0]))
            end_center = float(np.median(definition_values[:, -1]))
        elif threshold_statistic in {"mean", "discovery_route_midpoint"}:
            start_center = float(definition_values[:, 0].mean())
            end_center = float(definition_values[:, -1].mean())
        else:
            raise ValueError(
                "threshold_statistic must be mean, median, or discovery_route_midpoint"
            )
        change = end_center - start_center
        if abs(change) < min_change_nm:
            continue
        threshold = 0.5 * (start_center + end_center)
        initial_distance = None
        initial_progress_margin = None
        if initial_ca_nm is not None:
            initial_distance = float(np.linalg.norm(initial_ca_nm[i] - initial_ca_nm[j]))
            initial_progress_margin = (
                initial_distance - threshold if change < 0 else threshold - initial_distance
            )
            if (
                initial_progress_margin <= 0.0
                or initial_progress_margin < float(min_initial_margin_nm)
            ):
                continue
        crossing = []
        crossings_by_segment: list[int | None] = []
        for trace in values:
            hits = trace <= threshold if change < 0 else trace >= threshold
            first = _first_persistent_crossing(hits, min_crossing_persistence)
            if first is not None:
                crossing.append(first)
                crossings_by_segment.append(first)
            else:
                crossings_by_segment.append(None)
        stability = len(crossing) / len(values)
        support_count = len(crossing)
        required_event_support = (
            max(1, int(min_event_support_count))
            if min_event_support_count is not None
            else max(1, int(np.ceil(stable_fraction * len(values))))
        )
        if support_count < required_event_support:
            continue
        if (
            discovery_segment_index is not None
            and crossings_by_segment[int(discovery_segment_index)] is None
        ):
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
                "support_count": support_count,
                "support_segment_indices": [
                    index for index, value in enumerate(crossings_by_segment) if value is not None
                ],
                "median_crossing": median_crossing,
                "crossings": crossing,
                "crossings_by_segment": crossings_by_segment,
                "initial_distance_nm": initial_distance,
                "initial_progress_margin_nm": initial_progress_margin,
            }
        )
    candidates.sort(
        key=lambda row: (row["support_count"], abs(row["change_nm"])), reverse=True
    )
    route_specific = discovery_segment_index is not None or min_pair_support_count is not None
    eligible_pairs = []
    for left in candidates:
        for right in candidates:
            if {left["residue_i"], left["residue_j"]} == {right["residue_i"], right["residue_j"]}:
                continue
            pair = _ordered_pair_diagnostics(
                left,
                right,
                segment_count=len(segments),
                min_event_separation=min_event_separation,
            )
            if discovery_segment_index is not None and int(discovery_segment_index) not in pair[
                "ordered_segment_indices"
            ]:
                continue
            if route_specific:
                required_pair_support = max(1, int(min_pair_support_count or 1))
                if pair["pair_order_support_count"] < required_pair_support:
                    continue
                eligible_pairs.append((left, right, pair))
            else:
                if right["median_crossing"] - left["median_crossing"] < min_event_separation:
                    continue
                if pair["pair_order_stability"] < stable_fraction:
                    continue
                return [left, right], {
                    "candidate_count": len(candidates),
                    "top_candidates": candidates[:50],
                    "selected_pair_diagnostics": pair,
                }
    if eligible_pairs:
        # Prefer independently repeated routes, then stronger structural change.
        left, right, pair = max(
            eligible_pairs,
            key=lambda item: (
                item[2]["pair_order_support_count"],
                min(item[0]["support_count"], item[1]["support_count"]),
                abs(item[0]["change_nm"]) + abs(item[1]["change_nm"]),
            ),
        )
        pair["route_status"] = (
            "single_reference_route"
            if pair["pair_order_support_count"] == 1
            else "independently_supported_route"
        )
        return [left, right], {
            "candidate_count": len(candidates),
            "top_candidates": candidates[:50],
            "selected_pair_diagnostics": pair,
            "route_definition_segment_index": discovery_segment_index,
            "route_support_count": pair["pair_order_support_count"],
            "route_support_fraction": pair["pair_order_stability"],
            "route_status": pair["route_status"],
        }
    raise RuntimeError(
        f"No two supported distance events separated by {min_event_separation} physical frames; "
        f"retained {len(candidates)} candidates"
    )


def _ordered_pair_diagnostics(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    segment_count: int,
    min_event_separation: int,
) -> dict[str, Any]:
    """Measure event ordering on each design segment, not only median times."""
    left_crossings = list(left.get("crossings_by_segment", left.get("crossings", [])))
    right_crossings = list(right.get("crossings_by_segment", right.get("crossings", [])))
    if len(left_crossings) != segment_count or len(right_crossings) != segment_count:
        return {
            "paired_segment_coverage": 0.0,
            "pair_order_stability": 0.0,
            "pair_order_support_count": 0,
            "paired_segment_indices": [],
            "ordered_segment_indices": [],
            "ordered_gaps": [],
            "reason": "crossing arrays do not cover every design segment",
        }
    paired = [
        (index, int(right_time) - int(left_time))
        for index, (left_time, right_time) in enumerate(zip(left_crossings, right_crossings))
        if left_time is not None and right_time is not None
    ]
    gaps = [gap for _, gap in paired]
    ordered = [(index, gap) for index, gap in paired if gap >= int(min_event_separation)]
    return {
        "paired_segment_coverage": float(len(gaps) / segment_count),
        "pair_order_stability": float(len(ordered) / segment_count),
        "pair_order_support_count": len(ordered),
        "paired_segment_indices": [index for index, _ in paired],
        "ordered_segment_indices": [index for index, _ in ordered],
        "ordered_gaps": gaps,
        "minimum_required_gap": int(min_event_separation),
    }


def prepare_atlas_programs(cfg: dict[str, Any]) -> Path:
    reference = cfg["reference"]
    output = cfg["output"]
    protocol = cfg.get("benchmark_protocol", {})
    transition_mode = str(protocol.get("transition_mode", "legacy_normalized"))
    route_v3_1 = transition_mode == "last_exit_route_geometry"
    if route_v3_1 and str(protocol.get("version")) != "route_v3_1":
        raise ValueError("last_exit_route_geometry requires benchmark_protocol.version route_v3_1")
    topology = Path(reference["topology"]).expanduser().resolve()
    trajectories = [Path(path).expanduser().resolve() for path in reference["design_trajectories"]]
    output_dir = Path(output["directory"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    design_pca_dir = output_dir / ("design_pca_d1_d2" if route_v3_1 else "design_pca_r1_r2")
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
    initial_ca_nm = _load_initial_ca(start_pdb)
    width = float(reference.get("basin_half_width", 0.25))
    target_width = float(reference.get("target_basin_half_width", width))
    start_interval = (start_score - width, start_score + width)
    lower, upper = np.quantile(all_scores, [0.10, 0.90])
    if route_v3_1:
        endpoint_value = reference.get("route_endpoint_structure")
        if not endpoint_value:
            raise KeyError("reference.route_endpoint_structure is required for route_v3_1")
        endpoint_pdb = Path(endpoint_value).expanduser().resolve()
        target_center = float(pca.project_files(endpoint_pdb)[0, 0])
    else:
        endpoint_pdb = None
        target_center = float(
            upper if abs(upper - start_score) >= abs(lower - start_score) else lower
        )
    target_interval = (target_center - target_width, target_center + target_width)
    transition_target_interval = target_interval if route_v3_1 else (
        target_center - width,
        target_center + width,
    )
    middle = 0.5 * (start_score + target_center)
    intermediate_interval = (middle - width, middle + width)
    runtime_horizon = cfg.get("trajectory", {}).get("horizon")
    horizon = int(cfg.get("trajectory_horizon", runtime_horizon or 8))
    if (
        transition_mode in {"last_exit_fixed_lag", "last_exit_route_geometry"}
        and runtime_horizon is not None
        and horizon != int(runtime_horizon)
    ):
        raise ValueError(
            "trajectory_horizon and trajectory.horizon must match for route-v3 construction"
        )
    segments_by_trajectory: list[np.ndarray] = []
    segment_trajectories = []
    segment_source_indices: list[int] = []
    transition_path_reports: list[dict[str, Any]] = []
    discovery_segment_index: int | None = None
    model_step_ps: float | None = None
    nominal_challenge_report: dict[str, Any] | None = None
    if transition_mode == "last_exit_fixed_lag":
        model_step_ps = float(cfg["model"]["physical_lag_in_10ps"]) * 10.0
        ratio_cfg = protocol.get("duration_ratio", [0.80, 1.10])
        if len(ratio_cfg) != 2 or float(ratio_cfg[0]) <= 0 or float(ratio_cfg[0]) > float(
            ratio_cfg[1]
        ):
            raise ValueError("benchmark_protocol.duration_ratio must be [positive_min, max]")
        max_start_rmsd = protocol.get("max_start_ca_rmsd_nm")
        for source_index, (trajectory, values) in enumerate(zip(loaded, scores)):
            start_match = None
            start_rmsd = None
            if max_start_rmsd is not None:
                start_rmsd = _ca_rmsd_to_reference(
                    trajectory.xyz[:, ca_indices, :], initial_ca_nm
                )
                start_match = start_rmsd <= float(max_start_rmsd)
            result = _transition_path_segment(
                values,
                np.asarray(trajectory.time, dtype=float),
                start_interval,
                transition_target_interval,
                frames=horizon + 1,
                model_step_ps=model_step_ps,
                start_dwell_steps=float(protocol.get("start_dwell_model_steps", 0.5)),
                target_dwell_steps=float(protocol.get("target_dwell_model_steps", 0.5)),
                duration_ratio=(float(ratio_cfg[0]), float(ratio_cfg[1])),
                max_time_error_steps=float(protocol.get("max_sampling_time_error_steps", 0.1)),
                start_match_mask=start_match,
            )
            if result is None:
                continue
            indices, path_report = result
            path_report.update(
                {
                    "source_trajectory_index": source_index,
                    "source_trajectory": str(trajectories[source_index]),
                    "start_ca_rmsd_nm": (
                        None
                        if start_rmsd is None
                        else float(start_rmsd[path_report["start_exit_index"]])
                    ),
                }
            )
            segments_by_trajectory.append(indices)
            segment_trajectories.append(trajectory)
            segment_source_indices.append(source_index)
            transition_path_reports.append(path_report)
        requested_discovery = int(protocol.get("route_discovery_trajectory_index", 0))
        if requested_discovery in segment_source_indices:
            discovery_segment_index = segment_source_indices.index(requested_discovery)
        elif (
            bool(protocol.get("allow_first_eligible_discovery_fallback", False))
            and segment_source_indices
        ):
            discovery_segment_index = 0
        else:
            raise RuntimeError(
                "The predeclared route-discovery trajectory has no time-compatible "
                "transition path"
            )
    elif route_v3_1:
        requested_discovery = int(protocol.get("route_discovery_trajectory_index", 0))
        discovery_start_time_ps = protocol.get("route_discovery_start_time_ps")
        discovery_end_time_ps = protocol.get("route_discovery_end_time_ps")
        if discovery_start_time_ps is None or discovery_end_time_ps is None:
            raise KeyError(
                "route_v3_1 requires route_discovery_start_time_ps and "
                "route_discovery_end_time_ps"
            )
        analysis_frames = int(protocol.get("reference_route_analysis_frames", horizon + 1))
        max_start_rmsd = protocol.get("max_start_ca_rmsd_nm")
        for source_index, (trajectory, values) in enumerate(zip(loaded, scores)):
            start_rmsd = None
            start_match = None
            if max_start_rmsd is not None:
                start_rmsd = _ca_rmsd_to_reference(
                    trajectory.xyz[:, ca_indices, :], initial_ca_nm
                )
                start_match = start_rmsd <= float(max_start_rmsd)
            is_discovery = source_index == requested_discovery
            result = _transition_path_geometry_segment(
                values,
                np.asarray(trajectory.time, dtype=float),
                start_interval,
                transition_target_interval,
                analysis_frames=analysis_frames,
                start_dwell_ps=float(protocol.get("reference_start_dwell_ps", 200.0)),
                target_dwell_ps=float(protocol.get("reference_target_dwell_ps", 200.0)),
                start_match_mask=start_match,
                not_before_time_ps=(
                    float(discovery_start_time_ps) if is_discovery else None
                ),
                target_not_after_time_ps=(
                    float(discovery_end_time_ps) if is_discovery else None
                ),
            )
            if result is None:
                continue
            indices, path_report = result
            path_report.update(
                {
                    "source_trajectory_index": source_index,
                    "source_trajectory": str(trajectories[source_index]),
                    "source_role": (
                        str(reference.get("design_labels", [])[source_index])
                        if source_index < len(reference.get("design_labels", []))
                        else f"design_{source_index}"
                    ),
                    "start_ca_rmsd_nm": (
                        None
                        if start_rmsd is None
                        else float(start_rmsd[path_report["start_exit_index"]])
                    ),
                }
            )
            segments_by_trajectory.append(indices)
            segment_trajectories.append(trajectory)
            segment_source_indices.append(source_index)
            transition_path_reports.append(path_report)
        if requested_discovery not in segment_source_indices:
            raise RuntimeError(
                "The predeclared D1 route has no stable last-exit transition to the "
                "official endpoint basin"
            )
        discovery_segment_index = segment_source_indices.index(requested_discovery)
        discovery_report = transition_path_reports[discovery_segment_index]
        lag_mode = str(
            protocol.get(
                "lag_selection_mode", "reference_first_passage_target_compression"
            )
        )
        if lag_mode == "reference_first_passage_target_compression":
            lag_bounds = protocol.get("model_lag_bounds_in_10ps", [20, 210])
            if len(lag_bounds) != 2:
                raise ValueError("model_lag_bounds_in_10ps must contain two integers")
            selected_lag = _select_model_lag_in_10ps(
                float(discovery_report["reference_first_passage_ps"]),
                horizon=horizon,
                target_compression=float(protocol.get("target_nominal_compression", 2.0)),
                minimum=int(lag_bounds[0]),
                maximum=int(lag_bounds[1]),
            )
        elif lag_mode == "configured":
            selected_lag = int(cfg["model"]["physical_lag_in_10ps"])
        else:
            raise ValueError(
                "lag_selection_mode must be reference_first_passage_target_compression "
                "or configured"
            )
        cfg["model"]["physical_lag_in_10ps"] = selected_lag
        model_step_ps = float(selected_lag * 10)
        model_duration_ps = float(horizon * model_step_ps)
        nominal_compression = float(
            discovery_report["reference_first_passage_ps"] / model_duration_ps
        )
        compression_bounds = protocol.get("nominal_compression_factor", [1.0, 10.0])
        if (
            len(compression_bounds) != 2
            or float(compression_bounds[0]) <= 0.0
            or float(compression_bounds[0]) > float(compression_bounds[1])
        ):
            raise ValueError("nominal_compression_factor must be [positive_min, max]")
        if not float(compression_bounds[0]) <= nominal_compression <= float(
            compression_bounds[1]
        ):
            raise RuntimeError(
                f"Selected nominal compression {nominal_compression:.3f} is outside "
                f"[{float(compression_bounds[0]):.3f}, "
                f"{float(compression_bounds[1]):.3f}]"
            )
        nominal_challenge_report = {
            "lag_selection_mode": lag_mode,
            "selected_physical_lag_in_10ps": selected_lag,
            "model_step_ps": model_step_ps,
            "model_duration_ps": model_duration_ps,
            "reference_first_passage_ps": float(
                discovery_report["reference_first_passage_ps"]
            ),
            "reference_transition_path_ps": float(
                discovery_report["reference_transition_path_ps"]
            ),
            "nominal_compression": nominal_compression,
            "allowed_nominal_compression": [
                float(compression_bounds[0]),
                float(compression_bounds[1]),
            ],
            "physical_kinetics_claim": False,
        }
    elif transition_mode == "legacy_normalized":
        for source_index, (trajectory, values) in enumerate(zip(loaded, scores)):
            segments = _transition_segments(
                [values], start_interval, transition_target_interval, horizon + 1
            )
            if segments:
                segments_by_trajectory.append(segments[0])
                segment_trajectories.append(trajectory)
                segment_source_indices.append(source_index)
    else:
        raise ValueError(
            "benchmark_protocol.transition_mode must be legacy_normalized, "
            "last_exit_fixed_lag, or last_exit_route_geometry"
        )
    contact_cfg = cfg.get("contact_selection", {})
    if "min_event_separation_fraction" in contact_cfg:
        min_event_separation = max(
            1, int(np.ceil(horizon * float(contact_cfg["min_event_separation_fraction"])))
        )
    else:
        min_event_separation = int(contact_cfg.get("min_event_separation", 2))
    selected, report = _select_ordered_contacts(
        segment_trajectories,
        ca_indices,
        segments_by_trajectory,
        min_sequence_separation=int(contact_cfg.get("min_sequence_separation", 5)),
        min_change_nm=float(contact_cfg.get("min_change_nm", 0.15)),
        min_event_separation=min_event_separation,
        stable_fraction=float(contact_cfg.get("stable_fraction", 0.5)),
        max_event_frame=horizon - (2 if route_v3_1 else 1),
        initial_ca_nm=initial_ca_nm,
        min_initial_margin_nm=float(contact_cfg.get("min_initial_margin_nm", 0.0)),
        discovery_segment_index=discovery_segment_index,
        min_event_support_count=(
            None
            if transition_mode == "legacy_normalized"
            else int(contact_cfg.get("min_event_support_count", 1))
        ),
        min_pair_support_count=(
            None
            if transition_mode == "legacy_normalized"
            else int(contact_cfg.get("min_route_support_count", 1))
        ),
        threshold_statistic=str(contact_cfg.get("threshold_statistic", "mean")),
        min_crossing_persistence=int(contact_cfg.get("reference_crossing_persistence", 1)),
    )
    model_reference = model_path
    configured_root = cfg.get("project_root")
    if configured_root:
        try:
            model_reference = model_path.relative_to(
                Path(str(configured_root)).expanduser().resolve()
            )
        except ValueError:
            pass
    observables: dict[str, Any] = {
        "pc1": {"kind": "pc1", "pca_model": str(model_reference), "component": 0}
    }
    events = []
    event_half_width = max(
        1,
        int(
            round(
                horizon
                * float(protocol.get("event_window_padding_fraction", 1.0 / 8.0))
            )
        ),
    )
    event_names = (
        ("distance_A", "distance_B")
        if transition_mode in {"last_exit_fixed_lag", "last_exit_route_geometry"}
        else ("contact_A", "contact_B")
    )
    supporting_indices = report.get("selected_pair_diagnostics", {}).get(
        "ordered_segment_indices", []
    )
    event_window_source = str(protocol.get("event_window_source", "discovery_route"))
    if route_v3_1:
        window_segment_indices = [int(discovery_segment_index)]
    elif transition_mode == "last_exit_fixed_lag":
        if event_window_source == "discovery_route":
            window_segment_indices = [int(discovery_segment_index)]
        elif event_window_source == "supporting_routes":
            window_segment_indices = supporting_indices
        else:
            raise ValueError(
                "benchmark_protocol.event_window_source must be discovery_route "
                "or supporting_routes"
            )
    else:
        window_segment_indices = supporting_indices
    for name, row in zip(event_names, selected):
        observables[name] = {
            "kind": "residue_distance",
            "residue_i": row["residue_i"],
            "residue_j": row["residue_j"],
            "threshold_nm": row["threshold_nm"],
        }
        if window_segment_indices:
            crossings = [
                int(row["crossings_by_segment"][index])
                for index in window_segment_indices
                if row["crossings_by_segment"][index] is not None
            ]
        else:
            crossings = [int(value) for value in row["crossings"]]
        if not crossings:
            raise RuntimeError(f"No route-supported crossings available for {name}")
        contact_target_interval = (
            [0.0, row["threshold_nm"]]
            if int(row["target_contact"]) == 1
            else [row["threshold_nm"], 1000.0]
        )
        events.append(
            {
                "name": name,
                "observable": name,
                "target_interval": contact_target_interval,
                "physical_window": (
                    [1, horizon]
                    if route_v3_1
                    else [
                        max(1, min(crossings) - event_half_width),
                        min(horizon - 1, max(crossings) + event_half_width),
                    ]
                ),
                **(
                    {
                        "persistence_frames": int(
                            protocol.get("generated_event_persistence_frames", 2)
                        )
                    }
                    if route_v3_1
                    else {}
                ),
            }
        )
    terminal_mode = str(protocol.get("terminal_mode", "late_window"))
    if terminal_mode == "first_stable_hit_by_deadline":
        terminal_window = [1, horizon]
    elif terminal_mode == "final_frame":
        terminal_window = [horizon, horizon]
    elif terminal_mode == "late_window":
        terminal_window = [max(1, horizon - event_half_width), horizon]
    else:
        raise ValueError(
            "benchmark_protocol.terminal_mode must be late_window, final_frame, "
            "or first_stable_hit_by_deadline"
        )
    terminal = {
        "name": "terminal_basin",
        "observable": "pc1",
        "target_interval": list(target_interval),
        "physical_window": terminal_window,
        **(
            {
                "persistence_frames": int(
                    protocol.get("generated_target_persistence_frames", 2)
                )
            }
            if route_v3_1
            else {}
        ),
    }
    intermediate_center = int(round(horizon / 2))
    intermediate_half_width = max(1, int(round(horizon / 8)))
    design_labels = reference.get(
        "design_labels", [f"R{index + 1}" for index in range(len(trajectories))]
    )
    held_out_labels = reference.get(
        "held_out_labels",
        [
            f"R{len(trajectories) + index + 1}"
            for index, _ in enumerate(reference.get("held_out_trajectories", []))
        ],
    )
    held_out_diagnostic: dict[str, Any] = {
        "status": "not_configured",
        "transition_count": 0,
        "transition_paths": [],
    }
    if route_v3_1 and reference.get("held_out_trajectories"):
        held_out_paths = [
            Path(path).expanduser().resolve()
            for path in reference.get("held_out_trajectories", [])
        ]
        _, held_out_ca_indices, held_out_loaded = _load_design_trajectories(
            topology,
            held_out_paths,
            stride=int(reference.get("stride", 10)),
            max_frames_per_trajectory=reference.get("max_frames_per_trajectory"),
        )
        if len(held_out_ca_indices) != len(ca_indices):
            raise RuntimeError("Held-out topology does not match D1/D2 C-alpha count")
        held_out_reports = []
        for index, held_out_trajectory in enumerate(held_out_loaded):
            held_out_score = pca.project_trajectory(held_out_trajectory)[:, 0]
            held_out_rmsd = _ca_rmsd_to_reference(
                held_out_trajectory.xyz[:, held_out_ca_indices, :], initial_ca_nm
            )
            held_out_result = _transition_path_geometry_segment(
                held_out_score,
                np.asarray(held_out_trajectory.time, dtype=float),
                start_interval,
                transition_target_interval,
                analysis_frames=int(
                    protocol.get("reference_route_analysis_frames", horizon + 1)
                ),
                start_dwell_ps=float(protocol.get("reference_start_dwell_ps", 200.0)),
                target_dwell_ps=float(protocol.get("reference_target_dwell_ps", 200.0)),
                start_match_mask=(
                    held_out_rmsd <= float(protocol.get("max_start_ca_rmsd_nm", 0.30))
                ),
            )
            if held_out_result is None:
                continue
            _, held_out_report = held_out_result
            held_out_report.update(
                {
                    "source_trajectory": str(held_out_paths[index]),
                    "source_role": (
                        str(held_out_labels[index])
                        if index < len(held_out_labels)
                        else f"held_out_{index}"
                    ),
                    "start_ca_rmsd_nm": float(
                        held_out_rmsd[held_out_report["start_exit_index"]]
                    ),
                }
            )
            held_out_reports.append(held_out_report)
        held_out_diagnostic = {
            "status": "available" if held_out_reports else "unavailable",
            "reason": (
                None
                if held_out_reports
                else "H contains no comparable stable start-to-target transition"
            ),
            "transition_count": len(held_out_reports),
            "transition_paths": held_out_reports,
            "eligibility_gate": False,
        }

    tasks = {
        "endpoint": {"type": "terminal", "events": [terminal]},
        "ordered": {
            "type": "ordered",
            "allow_same_frame": False,
            "events": events,
            "terminal_event": terminal,
        },
    }
    if not route_v3_1:
        tasks["windowed"] = {
            "type": "windowed",
            "events": [
                {
                    "name": "intermediate_pc1",
                    "observable": "pc1",
                    "target_interval": list(intermediate_interval),
                    "physical_window": [
                        max(1, intermediate_center - intermediate_half_width),
                        min(horizon - 1, intermediate_center + intermediate_half_width),
                    ],
                }
            ],
            "terminal_event": terminal,
        }
    catalog = {
        "protocol_version": str(protocol.get("version", "legacy")),
        "design_split": {
            "task_design": list(design_labels),
            "held_out_evaluation": list(held_out_labels),
        },
        "observables": observables,
        "tasks": tasks,
        "selection": {
            "transition_mode": transition_mode,
            "protocol_parameters": dict(protocol),
            "trajectory_horizon": horizon,
            "model_step_ps": model_step_ps,
            "nominal_challenge": nominal_challenge_report,
            "replicate_roles": dict(reference.get("replicate_roles", {})),
            "start_pc1": start_score,
            "target_pc1": target_center,
            "route_endpoint_structure": (
                None if endpoint_pdb is None else str(endpoint_pdb)
            ),
            "start_interval": list(start_interval),
            "transition_target_interval": list(transition_target_interval),
            "target_interval": list(target_interval),
            "intermediate_interval": list(intermediate_interval),
            "transition_segment_count": len(segments_by_trajectory),
            "design_trajectory_count": len(trajectories),
            "transition_segment_source_indices": segment_source_indices,
            "transition_paths": transition_path_reports,
            "route_discovery_segment_index": discovery_segment_index,
            "route_discovery_source_trajectory_index": (
                None
                if discovery_segment_index is None
                else segment_source_indices[discovery_segment_index]
            ),
            "route_support_denominator": len(segments_by_trajectory),
            "held_out_diagnostic": held_out_diagnostic,
            "ordered_events": events,
            "terminal_event": terminal,
            "selected_contacts": selected,
            **report,
        },
    }
    # Protocol-specific intermediate twists may be stored with each task so a
    # frozen legacy runner can consume v2 catalogs without changing its global
    # configuration merge rules.
    for task in catalog["tasks"].values():
        if "failure_guidance_weight" in cfg.get("program", {}):
            task["failure_guidance_weight"] = float(
                cfg["program"]["failure_guidance_weight"]
            )
    catalog_path = output_dir / "duet_programs.yaml"
    with catalog_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(catalog, handle, sort_keys=False)
    with (output_dir / "selection_report.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(catalog["selection"], handle, sort_keys=False)
    return catalog_path
