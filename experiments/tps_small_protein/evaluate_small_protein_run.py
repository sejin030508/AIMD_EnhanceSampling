#!/usr/bin/env python3
"""Post-evaluate one completed small-protein ConfRover/SMC run.

Sampling outputs are never changed.  TICA/THP, heavy RMSD, PMF overlays,
TICA-DTW, CA step displacement, and conditional sampled-frame ETS are written
next to the production metrics under explicitly named files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import openmm as mm
from openmm import app, unit
import pyemma.coordinates as coor


ATOM37_NAMES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
ATOM37_INDEX = {name: index for index, name in enumerate(ATOM37_NAMES)}
BB_INDICES = np.asarray([ATOM37_INDEX[name] for name in ("N", "CA", "C")])
CA_INDEX = ATOM37_INDEX["CA"]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fit(moving: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (moving - moving_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.linalg.det(left @ right_t)])
    rotation = left @ correction @ right_t
    return rotation, moving_center, target_center


def aligned_rmsd(moving: np.ndarray, target: np.ndarray) -> float:
    rotation, moving_center, target_center = fit(moving, target)
    aligned = (moving - moving_center) @ rotation + target_center
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=-1))))


def backbone_rmsd(frame: np.ndarray, target: np.ndarray) -> float:
    residues = np.arange(frame.shape[0])
    rows = np.repeat(residues, len(BB_INDICES))
    atoms = np.tile(BB_INDICES, len(residues))
    return aligned_rmsd(frame[rows, atoms], target[rows, atoms])


def topology_atom37_map(topology: md.Topology, residue_count: int) -> list[tuple[int, int, int]]:
    mapping = []
    residues = list(topology.residues)
    if len(residues) != residue_count:
        raise ValueError(f"topology residue count {len(residues)} != {residue_count}")
    for atom in topology.atoms:
        if atom.element is None or atom.element.symbol.upper() == "H":
            continue
        if atom.name not in ATOM37_INDEX:
            raise ValueError(f"unsupported official heavy atom {atom.residue}:{atom.name}")
        mapping.append((atom.index, atom.residue.index, ATOM37_INDEX[atom.name]))
    return mapping


def heavy_rmsd(
    frame: np.ndarray,
    frame_mask: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    official_map: list[tuple[int, int, int]],
) -> tuple[float, int, bool]:
    points, references = [], []
    all_official = True
    for _, residue, atom in official_map:
        available = bool(frame_mask[residue, atom] and target_mask[residue, atom])
        all_official &= available
        if available:
            points.append(frame[residue, atom])
            references.append(target[residue, atom])
    if len(points) < 3:
        return float("nan"), len(points), False
    return (
        aligned_rmsd(np.asarray(points), np.asarray(references)),
        len(points),
        all_official,
    )


class OfficialTICA:
    def __init__(self, manifest: dict[str, Any]) -> None:
        source_files = {Path(row["path"]).name: Path(row["path"]) for row in manifest["official_source_files"]}
        self.folded = source_files["folded.pdb"]
        self.model_path = Path(manifest["tica_model"])
        self.topology_trajectory = md.load(str(self.folded))
        self.topology = self.topology_trajectory.topology
        self.feature = coor.featurizer(str(self.folded))
        self.feature.add_backbone_torsions(cossin=True)
        self.model = joblib.load(self.model_path)
        template = md.load(manifest["minimized_target_allatom"])
        if template.n_atoms != self.topology.n_atoms:
            raise ValueError("minimized target and official folded topology differ")
        self.template_xyz_nm = template.xyz[0].copy()
        self.mapping = topology_atom37_map(
            self.topology, int(manifest["model_length"])
        )

    def project(self, frames_a: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames_a = np.asarray(frames_a)
        masks = np.asarray(masks, dtype=bool)
        output = np.full((len(frames_a), 2), np.nan, dtype=float)
        usable = np.ones(len(frames_a), dtype=bool)
        xyz = np.repeat(self.template_xyz_nm[None], len(frames_a), axis=0)
        for topology_atom, residue, atom37 in self.mapping:
            available = masks[:, residue, atom37]
            # TICA only uses backbone torsions, but preserve every available
            # generated heavy atom. Missing non-backbone atoms do not matter.
            xyz[available, topology_atom] = frames_a[available, residue, atom37] / 10.0
        for residue in range(frames_a.shape[1]):
            usable &= np.all(masks[:, residue, BB_INDICES], axis=1)
        if np.any(usable):
            trajectory = md.Trajectory(xyz[usable], self.topology)
            features = self.feature.transform(trajectory)
            transformed = np.asarray(self.model.transform(features))
            output[usable] = transformed[:, :2]
        return output, usable


def ca_step_displacement(frames: np.ndarray, masks: np.ndarray) -> list[float | None]:
    values: list[float | None] = [0.0]
    for left, right, left_mask, right_mask in zip(
        frames[:-1], frames[1:], masks[:-1], masks[1:]
    ):
        common = left_mask[:, CA_INDEX] & right_mask[:, CA_INDEX]
        if int(np.sum(common)) < 3:
            values.append(None)
        else:
            values.append(aligned_rmsd(right[common, CA_INDEX], left[common, CA_INDEX]))
    return values


def dtw_normalized(left: np.ndarray, right: np.ndarray) -> float:
    n, m = len(left), len(right)
    cost = np.full((n + 1, m + 1), np.inf)
    length = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [(cost[i - 1, j], length[i - 1, j]),
                       (cost[i, j - 1], length[i, j - 1]),
                       (cost[i - 1, j - 1], length[i - 1, j - 1])]
            previous_cost, previous_length = min(choices, key=lambda item: item[0])
            cost[i, j] = previous_cost + float(np.linalg.norm(left[i - 1] - right[j - 1]))
            length[i, j] = previous_length + 1
    return float(cost[n, m] / length[n, m])


def trace_pre_final_lineages(ancestry: list[list[int]], path_count: int) -> list[list[int]]:
    horizon = len(ancestry)
    lineages = []
    for final_index in range(path_count):
        current = final_index
        reverse = [current]
        # The last map is the final post-evaluation resampling, whereas the
        # stored population is pre-final. Trace only maps through T-1.
        for mapping in reversed(ancestry[:-1]):
            current = int(mapping[current])
            reverse.append(current)
        lineages.append(list(reversed(reverse)))
    if any(len(row) != horizon for row in lineages):
        raise AssertionError("lineage reconstruction length mismatch")
    return lineages


class SampledFrameEnergy:
    """Official potential with generated heavy atoms held exactly fixed."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        source_files = {Path(row["path"]).name: Path(row["path"]) for row in manifest["official_source_files"]}
        official_root = Path(manifest["tica_model"]).parents[2]
        forcefield = app.ForceField(
            str(official_root / "data" / "protein.ff14SBonlysc.xml"),
            "implicit/gbn2.xml",
        )
        pdb = app.PDBFile(str(source_files["folded.pdb"]))
        self.pdb = pdb
        self.system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            # The sampled-frame diagnostic freezes every generated heavy atom
            # by setting its mass to zero.  OpenMM forbids constraints that
            # involve a massless particle, so this evaluation-only system uses
            # the same force-field energy terms without holonomic constraints.
            # This restores the corresponding bond terms and lets only the
            # template hydrogens relax; it does not alter sampling or rewards.
            constraints=None,
            ewaldErrorTolerance=0.0005,
        )
        self.heavy = []
        self.mapping = []
        for atom in pdb.topology.atoms():
            is_heavy = atom.element is not None and atom.element.symbol.upper() != "H"
            if is_heavy:
                if atom.name not in ATOM37_INDEX:
                    raise ValueError(f"unrepresentable heavy atom {atom.residue}:{atom.name}")
                self.heavy.append(atom.index)
                self.mapping.append((atom.index, atom.residue.index, ATOM37_INDEX[atom.name]))
                self.system.setParticleMass(atom.index, 0.0 * unit.dalton)
        template = app.PDBFile(manifest["minimized_target_allatom"])
        self.template_a = np.asarray(
            template.positions.value_in_unit(unit.angstrom), dtype=float
        )
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        self.simulation = app.Simulation(pdb.topology, self.system, self.integrator)

    def energy(self, frame: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        heavy_frame, heavy_template, topology_indices = [], [], []
        for topology_atom, residue, atom37 in self.mapping:
            # ConfRover commonly omits terminal OXT.  It is not a generated
            # coordinate, so retain the aligned minimized-target coordinate
            # for that atom and freeze it with all other heavy atoms.  Every
            # heavy coordinate actually emitted by the model remains exact.
            if not bool(mask[residue, atom37]):
                continue
            topology_indices.append(topology_atom)
            heavy_frame.append(frame[residue, atom37])
            heavy_template.append(self.template_a[topology_atom])
        if len(topology_indices) < 3:
            raise ValueError("fewer than three generated heavy atoms available for ETS alignment")
        heavy_frame_array = np.asarray(heavy_frame)
        rotation, center, target_center = fit(np.asarray(heavy_template), heavy_frame_array)
        positions_a = (self.template_a - center) @ rotation + target_center
        positions_a[np.asarray(topology_indices)] = heavy_frame_array
        self.simulation.context.setPositions(positions_a * unit.angstrom)
        self.simulation.minimizeEnergy()
        state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        after_a = np.asarray(state.getPositions().value_in_unit(unit.angstrom))
        displacement = float(
            np.max(np.linalg.norm(after_a[np.asarray(topology_indices)] - heavy_frame_array, axis=1))
        )
        if displacement > 1.0e-5:
            raise RuntimeError(f"fixed heavy atoms moved by {displacement:.3e} A")
        energy = float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        )
        return energy, displacement


def plot_overlay(
    output: Path,
    manifest: dict[str, Any],
    tica: np.ndarray,
    target: np.ndarray,
) -> None:
    pmf = np.load(manifest["pmf"])
    xs = np.load(manifest["pmf_xs"])
    ys = np.load(manifest["pmf_ys"])
    field = pmf if pmf.shape == (len(xs), len(ys)) else pmf.T
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    contour = ax.contourf(xs, ys, field.T, levels=30, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="provided PMF value")
    for path_index, curve in enumerate(tica):
        usable = np.isfinite(curve).all(axis=1)
        ax.plot(curve[usable, 0], curve[usable, 1], marker="o", markersize=2,
                linewidth=1, alpha=0.75, label=f"retained path {path_index}")
    ax.scatter([target[0]], [target[1]], marker="*", s=130, c="red", label="folded target")
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    ax.set_title(f"{manifest['protein']} generated paths on supplied PMF")
    ax.legend(fontsize=7)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluate(run_dir: Path, prepared_dir: Path) -> dict[str, Any]:
    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    base_metrics = json.loads((run_dir / "metrics.json").read_text())
    population = np.load(run_dir / "pre_final_population_atom37.npz")
    trajectories = np.asarray(population["trajectories_atom37_a"], dtype=float)
    masks = np.asarray(population["trajectories_atom37_mask"], dtype=bool)
    weights = np.asarray(
        json.loads((run_dir / "pre_final_normalized_weights.json").read_text()),
        dtype=float,
    )
    weights = weights / weights.sum()
    valid = np.asarray(base_metrics["per_path_valid"], dtype=bool)
    reference = np.load(prepared_dir / "whole_backbone_reference_atom37.npz")
    target = np.asarray(reference["reference_atom37_a"][0], dtype=float)
    target_mask = np.asarray(reference["reference_atom37_mask"][0], dtype=bool)

    target_tica = np.asarray(manifest["tica"]["target_first_two"], dtype=float)
    tica_error = None
    try:
        official_tica = OfficialTICA(manifest)
        official_map = official_tica.mapping
        flattened = trajectories.reshape(-1, *trajectories.shape[2:])
        flattened_masks = masks.reshape(-1, *masks.shape[2:])
        tica_flat, tica_usable_flat = official_tica.project(flattened, flattened_masks)
        tica = tica_flat.reshape(trajectories.shape[0], trajectories.shape[1], 2)
        tica_usable = tica_usable_flat.reshape(trajectories.shape[:2])
    except Exception as error:
        # TICA is evaluation-only: an unavailable model must not erase a
        # completed generation or suppress the remaining structural metrics.
        tica_error = repr(error)
        source_files = {
            Path(row["path"]).name: Path(row["path"])
            for row in manifest["official_source_files"]
        }
        folded_topology = md.load(str(source_files["folded.pdb"])).topology
        official_map = topology_atom37_map(folded_topology, int(manifest["model_length"]))
        tica = np.full((*trajectories.shape[:2], 2), np.nan, dtype=float)
        tica_usable = np.zeros(trajectories.shape[:2], dtype=bool)
    tica_distance = np.linalg.norm(tica - target_tica[None, None], axis=-1)
    hits = np.isfinite(tica_distance) & (tica_distance < 0.75)
    final_hit = hits[:, -1]
    anytime_hit = np.any(hits[:, 1:], axis=1)
    valid_final_hit = valid & final_hit
    valid_anytime_hit = valid & anytime_hit

    bb = np.asarray([[backbone_rmsd(frame, target) for frame in path] for path in trajectories])
    heavy = np.zeros(trajectories.shape[:2], dtype=float)
    heavy_counts = np.zeros(trajectories.shape[:2], dtype=int)
    heavy_is_official = np.zeros(trajectories.shape[:2], dtype=bool)
    for i in range(len(trajectories)):
        for j in range(len(trajectories[i])):
            heavy[i, j], heavy_counts[i, j], heavy_is_official[i, j] = heavy_rmsd(
                trajectories[i, j], masks[i, j], target, target_mask, official_map
            )

    ancestry = json.loads((run_dir / "outer_ancestry.json").read_text())
    lineages = trace_pre_final_lineages(ancestry, len(trajectories))
    candidate_success = np.flatnonzero(valid_final_hit)
    unique_success = []
    seen: set[tuple[tuple[int, ...], str]] = set()
    for index in candidate_success:
        digest = hashlib.sha256(
            trajectories[index].tobytes()
            + masks[index].tobytes()
            + population["trajectories_aatype"][index].tobytes()
        ).hexdigest()
        key = (tuple(lineages[index]), digest)
        if key not in seen:
            seen.add(key)
            unique_success.append(int(index))
    dtw_pairs = []
    for offset, left in enumerate(unique_success):
        for right in unique_success[offset + 1:]:
            dtw_pairs.append(
                {
                    "left_path": left,
                    "right_path": right,
                    "normalized_tica_dtw": dtw_normalized(tica[left], tica[right]),
                }
            )

    first_hit = []
    ca_displacement = []
    for path_index, path_hits in enumerate(hits):
        generated_hits = np.flatnonzero(path_hits[1:])
        first_hit.append(int(generated_hits[0] + 1) if len(generated_hits) else None)
        ca_displacement.append(ca_step_displacement(trajectories[path_index], masks[path_index]))

    energy_rows = []
    failed_energy_evaluations = 0
    energy_evaluator_error = None
    if len(candidate_success):
        try:
            evaluator = SampledFrameEnergy(manifest)
        except Exception as error:
            energy_evaluator_error = repr(error)
            failed_energy_evaluations = int(len(candidate_success))
            energy_rows.extend(
                {
                    "path_index": int(path_index),
                    "status": "failed",
                    "error": energy_evaluator_error,
                }
                for path_index in candidate_success
            )
        else:
            for path_index in candidate_success:
                path_energies = []
                max_heavy_displacement = 0.0
                for frame, mask in zip(trajectories[path_index], masks[path_index]):
                    try:
                        energy, displacement = evaluator.energy(frame, mask)
                        path_energies.append(energy)
                        max_heavy_displacement = max(max_heavy_displacement, displacement)
                    except Exception as error:
                        failed_energy_evaluations += 1
                        energy_rows.append(
                            {"path_index": int(path_index), "status": "failed", "error": repr(error)}
                        )
                        path_energies = []
                        break
                if path_energies:
                    energy_rows.append(
                        {
                            "path_index": int(path_index),
                            "status": "complete",
                            "sampled_frame_max_potential_kj_mol": float(max(path_energies)),
                            "frame_potential_kj_mol": path_energies,
                            "max_fixed_heavy_atom_displacement_a": max_heavy_displacement,
                        }
                    )
    completed_ets = [
        row["sampled_frame_max_potential_kj_mol"]
        for row in energy_rows if row["status"] == "complete"
    ]

    diagnostics = {
        "tica_first_two": tica.tolist(),
        "tica_projection_usable": tica_usable.tolist(),
        "tica_distance_to_folded_target": tica_distance.tolist(),
        "target_hit": hits.tolist(),
        "first_hit_frame_including_x0_as_zero": first_hit,
        "ca_consecutive_kabsch_rmsd_a": ca_displacement,
        "whole_backbone_rmsd_a": bb.tolist(),
        "heavy_atom_rmsd_a": heavy.tolist(),
        "heavy_atom_common_counts": heavy_counts.tolist(),
        "heavy_atom_metric_is_official_full_mask": heavy_is_official.tolist(),
        "pre_final_lineages": lineages,
        "exact_duplicate_deduplicated_success_indices": unique_success,
        "tica_dtw_pairs": dtw_pairs,
        "energy": energy_rows,
    }
    write_json(run_dir / "small_protein_frame_diagnostics.json", diagnostics)
    np.savez_compressed(
        run_dir / "tica_projection.npz",
        tica_first_two=tica,
        distance_to_folded_target=tica_distance,
        target_hit=hits,
        usable=tica_usable,
    )
    pmf_overlay_error = None
    if tica_error is None:
        try:
            plot_overlay(run_dir / "tica_pmf_overlay.png", manifest, tica, target_tica)
        except Exception as error:
            pmf_overlay_error = repr(error)

    final_heavy = heavy[:, -1]
    official_final_mask = heavy_is_official[:, -1]
    metric_name = (
        "official_all-heavy-atom_Kabsch_RMSD"
        if np.all(official_final_mask)
        else "common-heavy-atom_Kabsch_RMSD"
    )
    result = {
        "status": "complete",
        "protein": manifest["protein"],
        "method": base_metrics["method"],
        "seed": int(base_metrics["seed"]),
        "stride_in_10ps": int(base_metrics["stride_in_10ps"]),
        "horizon": int(base_metrics["horizon"]),
        "population_size": int(len(trajectories)),
        "pre_final_normalized_weights": weights.tolist(),
        "whole_backbone_rmsd_unit": "Angstrom",
        "weighted_final_backbone_rmsd_a": float(np.dot(weights, bb[:, -1])),
        "best_valid_final_backbone_rmsd_a": (
            float(np.min(bb[valid, -1])) if np.any(valid) else None
        ),
        "heavy_rmsd_metric": metric_name,
        "weighted_final_heavy_rmsd_a": float(np.dot(weights, final_heavy)),
        "best_valid_final_heavy_rmsd_a": (
            float(np.min(final_heavy[valid])) if np.any(valid) else None
        ),
        "official_target_heavy_atom_count": int(len(official_map)),
        "per_path_final_common_heavy_atom_count": heavy_counts[:, -1].tolist(),
        "weighted_thp": (
            float(np.dot(weights, final_hit.astype(float))) if tica_error is None else None
        ),
        "weighted_valid_thp": (
            float(np.dot(weights, valid_final_hit.astype(float))) if tica_error is None else None
        ),
        "unweighted_thp": float(np.mean(final_hit)) if tica_error is None else None,
        "unweighted_valid_thp": (
            float(np.mean(valid_final_hit)) if tica_error is None else None
        ),
        "returned_target_hit_count": (
            int(np.sum(final_hit)) if tica_error is None else None
        ),
        "returned_valid_target_hit_count": (
            int(np.sum(valid_final_hit)) if tica_error is None else None
        ),
        "has_valid_target_path": (
            bool(np.any(valid_final_hit)) if tica_error is None else None
        ),
        "valid_anytime_hit_count": (
            int(np.sum(valid_anytime_hit)) if tica_error is None else None
        ),
        "valid_anytime_hit_fraction": (
            float(np.mean(valid_anytime_hit)) if tica_error is None else None
        ),
        "tica_evaluation_error": tica_error,
        "tica_metrics_na_reason": (
            None if tica_error is None else "official TICA projection failed; generation retained"
        ),
        "pmf_overlay_error": pmf_overlay_error,
        "invalid_paths_remain_in_denominator": True,
        "target_hit_definition": "first-two official TICA Euclidean distance < 0.75",
        "tica_target_self_distance": float(manifest["tica"]["target_self_distance"]),
        "path_diversity_normalized_tica_dtw_mean": (
            float(np.mean([row["normalized_tica_dtw"] for row in dtw_pairs]))
            if dtw_pairs else None
        ),
        "path_diversity_normalized_tica_dtw_median": (
            float(np.median([row["normalized_tica_dtw"] for row in dtw_pairs]))
            if dtw_pairs else None
        ),
        "path_diversity_evaluated_unique_valid_target_paths": len(unique_success),
        "path_diversity_na_reason": (
            None if len(unique_success) >= 2
            else "fewer than two distinct valid final target-reaching retained paths"
        ),
        "reference_path_coverage": None,
        "reference_path_coverage_na_reason": manifest["reference_path_coverage_na_reason"],
        "sampled_frame_ets_kj_mol_mean": (
            float(np.mean(completed_ets)) if completed_ets else None
        ),
        "sampled_frame_ets_kj_mol_median": (
            float(np.median(completed_ets)) if completed_ets else None
        ),
        "sampled_frame_ets_evaluated_path_count": len(completed_ets),
        "sampled_frame_ets_failed_energy_evaluation_count": failed_energy_evaluations,
        "sampled_frame_ets_evaluator_error": energy_evaluator_error,
        "sampled_frame_ets_na_reason": (
            None if completed_ets
            else (
                (
                    "official TICA projection failed; target-reaching paths undefined"
                    if tica_error is not None
                    else "no valid final target-reaching path"
                )
                if not len(candidate_success)
                else "energy evaluation failed for every qualifying path"
            )
        ),
        "sampled_frame_ets_interpretation": (
            "Maximum potential over saved frames only; not a physical transition-state energy"
        ),
        "sampled_frame_ets_protocol": (
            "official protein.ff14SBonlysc + implicit/gbn2 potential; "
            "evaluation-only unconstrained system; generated heavy atoms massless/fixed; "
            "official heavy atoms absent from the generated mask (normally terminal OXT) "
            "retained from the aligned minimized-target template and fixed; template hydrogens minimized"
        ),
        "decoder_nfe": int(base_metrics["decoder_nfe"]),
        "wall_clock_s": float(base_metrics["wall_clock_s"]),
        "whole_path_valid_fraction": float(base_metrics["valid_path_fraction"]),
        "base_metrics_file": str(run_dir / "metrics.json"),
        "manifest_file": str(prepared_dir / "manifest.json"),
    }
    write_json(run_dir / "small_protein_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.run_dir.resolve(), args.prepared_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
