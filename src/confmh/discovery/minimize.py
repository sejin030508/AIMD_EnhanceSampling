from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from confmh.discovery.structure import kabsch_rmsd, read_pdb, sequence, sequence_alignment_pairs


def minimize_pdb(
    input_pdb: str | Path,
    output_pdb: str | Path,
    *,
    max_iterations: int = 500,
    restraint_k_kj_mol_nm2: float = 1000.0,
) -> dict:
    """PDBFixer + restrained implicit-solvent OpenMM minimization."""
    import numpy as np
    from openmm import CustomExternalForce, LangevinMiddleIntegrator
    from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, Simulation
    from openmm.unit import kelvin, kilojoule_per_mole, nanometer, picosecond
    from pdbfixer import PDBFixer

    input_pdb, output_pdb = Path(input_pdb).resolve(), Path(output_pdb).resolve()
    fixer = PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = forcefield.createSystem(fixer.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", float(restraint_k_kj_mol_nm2) * kilojoule_per_mole / nanometer**2)
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)
    for atom, position in zip(fixer.topology.atoms(), fixer.positions):
        if atom.name in {"N", "CA", "C", "O"}:
            restraint.addParticle(atom.index, position.value_in_unit(nanometer))
    system.addForce(restraint)
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picosecond)
    simulation = Simulation(fixer.topology, system, integrator)
    simulation.context.setPositions(fixer.positions)
    before = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    simulation.minimizeEnergy(maxIterations=int(max_iterations))
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    after = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, state.getPositions(), handle, keepIds=True)
    original, minimized = read_pdb(input_pdb), read_pdb(output_pdb)
    pairs = sequence_alignment_pairs(sequence(original), sequence(minimized))
    ca_pairs = [(i, j) for i, j in pairs if "CA" in original[i].atoms and "CA" in minimized[j].atoms]
    pre = np.stack([original[i].atoms["CA"] for i, _ in ca_pairs])
    post = np.stack([minimized[j].atoms["CA"] for _, j in ca_pairs])
    return {
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "energy_before_kj_mol": float(before),
        "energy_after_kj_mol": float(after),
        "energy_decrease_kj_mol": float(before - after),
        "ca_rmsd_change_a": kabsch_rmsd(pre, post),
        "success": True,
    }


def minimize_candidates(
    candidate_paths: Iterable[str | Path], output_dir: str | Path, *, max_iterations: int = 500
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidate_paths:
        candidate = Path(candidate).expanduser().resolve()
        try:
            rows.append(
                minimize_pdb(candidate, output_dir / candidate.name, max_iterations=max_iterations)
            )
        except Exception as exc:
            rows.append({"input_pdb": str(candidate), "success": False, "error": repr(exc)})
    summary_path = output_dir / "minimization_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return summary_path


def hit_candidates_from_csv(path: str | Path, hit_column: str = "valid_hit_3a") -> list[Path]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Path(row["candidate_pdb"])
        for row in rows
        if str(row.get(hit_column, "")).lower() in {"true", "1", "yes"}
    ]
