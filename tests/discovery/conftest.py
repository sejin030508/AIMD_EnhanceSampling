from __future__ import annotations

from pathlib import Path


def write_backbone(path: Path, ca_xyz: list[tuple[float, float, float]]) -> Path:
    lines = []
    serial = 1
    for index, (x, y, z) in enumerate(ca_xyz, start=1):
        atoms = {
            "N": (x - 1.25, y + 0.40, z),
            "CA": (x, y, z),
            "C": (x + 1.30, y + 0.35, z),
            "O": (x + 1.65, y + 1.45, z),
        }
        for atom, (ax, ay, az) in atoms.items():
            element = atom[0]
            lines.append(
                f"ATOM  {serial:5d} {atom:^4s} ALA A{index:4d}    "
                f"{ax:8.3f}{ay:8.3f}{az:8.3f}  1.00  0.00          {element:>2s}  "
            )
            serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
