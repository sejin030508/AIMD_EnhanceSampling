#!/usr/bin/env python3
"""Extract the exact Phase-B author-chain constructs with PyMOL.

Run from the repository root with:
  /Applications/PyMOL.app/Contents/bin/pymol -cq scripts/duet/prepare_phase_b_structures_pymol.py
"""

from pathlib import Path

from pymol import cmd


ROOT = Path.cwd()
RAW = ROOT / "data/phase_b_pockets/raw"
PREPARED = ROOT / "data/phase_b_pockets/prepared"


def extract(source: Path, destination: Path, *, first: int, last: int) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd.reinitialize()
    cmd.load(str(source), "source")
    selection = f"source and polymer.protein and chain A and resi {first}-{last}"
    cmd.create("construct", selection, 1, 1)
    if cmd.count_atoms("construct") == 0:
        raise RuntimeError(f"Empty author-chain selection: {source} {selection}")
    cmd.remove("construct and hydro")
    cmd.remove("construct and not alt ''+A")
    cmd.alter("construct", "alt='' ")
    cmd.sort("construct")
    cmd.save(str(destination), "construct", state=1)
    ca_residues = {atom.resi for atom in cmd.get_model("construct and name CA").atom}
    print(
        f"prepared={destination} atoms={cmd.count_atoms('construct')} "
        f"ca_residues={len(ca_residues)} author_range={first}-{last}"
    )


extract(
    RAW / "7KIC.cif",
    PREPARED / "prmt5/source_numbering/7KIC_A_294_637_start.pdb",
    first=294,
    last=637,
)
extract(
    RAW / "6UXY.cif",
    PREPARED / "prmt5/source_numbering/6UXY_A_294_637_holo.pdb",
    first=294,
    last=637,
)
extract(
    RAW / "6UXX.cif",
    PREPARED / "prmt5/source_numbering/6UXX_A_294_637_holo.pdb",
    first=294,
    last=637,
)
extract(
    RAW / "AF-Q96LA8-F1-model_v4.pdb",
    PREPARED / "prmt6/source_numbering/AF-Q96LA8-F1-model_v4_53_375_start.pdb",
    first=53,
    last=375,
)
extract(
    RAW / "6W6D.cif",
    PREPARED / "prmt6/source_numbering/6W6D_A_53_375_holo.pdb",
    first=53,
    last=375,
)

cmd.quit()
