from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


def _kabsch_align(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Align (..., atoms, 3) coordinates to a fixed reference."""
    x = np.asarray(coords, dtype=float)
    ref = np.asarray(reference, dtype=float)
    original_shape = x.shape
    flat = x.reshape((-1,) + x.shape[-2:])
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    aligned = np.empty_like(flat)
    for idx, frame in enumerate(flat):
        center = frame.mean(axis=0, keepdims=True)
        centered = frame - center
        covariance = centered.T @ ref_centered
        u, _, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        aligned[idx] = centered @ rotation + ref.mean(axis=0, keepdims=True)
    return aligned.reshape(original_shape)


@dataclass
class PCACV:
    reference_ca_nm: np.ndarray
    mean_flat: np.ndarray
    components: np.ndarray
    score_mean: np.ndarray
    score_scale: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "PCACV":
        data = np.load(path)
        return cls(
            reference_ca_nm=np.asarray(data["reference_ca_nm"]),
            mean_flat=np.asarray(data["mean_flat"]),
            components=np.asarray(data["components"]),
            score_mean=np.asarray(data["score_mean"]),
            score_scale=np.asarray(data["score_scale"]),
        )

    def save(self, path: str | Path, explained_variance_ratio: np.ndarray | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reference_ca_nm": self.reference_ca_nm,
            "mean_flat": self.mean_flat,
            "components": self.components,
            "score_mean": self.score_mean,
            "score_scale": self.score_scale,
        }
        if explained_variance_ratio is not None:
            payload["explained_variance_ratio"] = explained_variance_ratio
        np.savez_compressed(path, **payload)
        return path

    @property
    def dimension(self) -> int:
        return int(self.components.shape[0])

    def project_ca(self, ca_nm: np.ndarray) -> np.ndarray:
        ca_nm = np.asarray(ca_nm, dtype=float)
        if ca_nm.shape[-2:] != self.reference_ca_nm.shape:
            raise ValueError(
                f"Expected {self.reference_ca_nm.shape[0]} CA atoms; got shape {ca_nm.shape}"
            )
        aligned = _kabsch_align(ca_nm, self.reference_ca_nm)
        flat = aligned.reshape((-1, aligned.shape[-2] * 3))
        raw = (flat - self.mean_flat) @ self.components.T
        standardized = (raw - self.score_mean) / self.score_scale
        return standardized.reshape(ca_nm.shape[:-2] + (self.dimension,))

    def project_trajectory(self, trajectory) -> np.ndarray:
        ca_indices = trajectory.topology.select("name CA")
        if len(ca_indices) != self.reference_ca_nm.shape[0]:
            raise ValueError(
                f"Generated topology has {len(ca_indices)} CA atoms; reference has "
                f"{self.reference_ca_nm.shape[0]}"
            )
        return self.project_ca(trajectory.xyz[:, ca_indices, :])

    def project_files(self, topology_pdb: str | Path, trajectory: str | Path | None = None) -> np.ndarray:
        import mdtraj as md

        if trajectory is None:
            traj = md.load(str(topology_pdb))
        else:
            traj = md.load(str(trajectory), top=str(topology_pdb))
        return self.project_trajectory(traj)


def _iter_protein_chunks(
    topology_pdb: Path,
    trajectories: Iterable[Path],
    *,
    protein_selection: str,
    stride: int,
    burn_in_frames: int,
    chunk_size: int,
    max_frames_per_trajectory: int | None,
):
    import mdtraj as md

    topology_frame = md.load(str(topology_pdb))
    protein_indices = topology_frame.topology.select(protein_selection)
    if len(protein_indices) == 0:
        raise ValueError(f"Selection matched no atoms: {protein_selection}")
    reference = topology_frame.atom_slice(protein_indices)
    ca_indices = reference.topology.select("name CA")
    if len(ca_indices) == 0:
        raise ValueError("Reference topology contains no CA atoms")

    for trajectory_index, trajectory_path in enumerate(trajectories):
        yielded = 0
        skipped = 0
        source_frame = 0
        for chunk in md.iterload(
            str(trajectory_path),
            top=str(topology_pdb),
            atom_indices=protein_indices,
            stride=stride,
            chunk=chunk_size,
        ):
            chunk_start = source_frame
            source_frame += len(chunk) * stride
            if skipped < burn_in_frames:
                remove = min(len(chunk), burn_in_frames - skipped)
                chunk = chunk[remove:]
                chunk_start += remove * stride
                skipped += remove
            if len(chunk) == 0:
                continue
            if max_frames_per_trajectory is not None:
                remaining = max_frames_per_trajectory - yielded
                if remaining <= 0:
                    break
                chunk = chunk[:remaining]
            chunk.superpose(reference, atom_indices=ca_indices, ref_atom_indices=ca_indices)
            frame_indices = chunk_start + np.arange(len(chunk)) * stride
            yield trajectory_index, trajectory_path, frame_indices, chunk, reference, ca_indices
            yielded += len(chunk)
            if max_frames_per_trajectory is not None and yielded >= max_frames_per_trajectory:
                break


def fit_reference_pca(
    *,
    topology_pdb: str | Path,
    trajectories: list[str | Path],
    output_dir: str | Path,
    protein_selection: str = "protein and chainid 0",
    stride: int = 1,
    burn_in_frames: int = 0,
    chunk_size: int = 1000,
    max_frames_per_trajectory: int | None = None,
    n_components: int = 2,
    n_seed_frames: int = 20,
) -> Path:
    """Fit aligned C-alpha PCA and build an empirical reference oracle."""
    topology_pdb = Path(topology_pdb).resolve()
    trajectories = [Path(path).resolve() for path in trajectories]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not topology_pdb.exists():
        raise FileNotFoundError(topology_pdb)
    missing = [path for path in trajectories if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trajectories: {missing}")

    ca_blocks: list[np.ndarray] = []
    source_traj: list[int] = []
    source_frame: list[int] = []
    reference_ca = None
    iterator = _iter_protein_chunks(
        topology_pdb,
        trajectories,
        protein_selection=protein_selection,
        stride=stride,
        burn_in_frames=burn_in_frames,
        chunk_size=chunk_size,
        max_frames_per_trajectory=max_frames_per_trajectory,
    )
    for trajectory_index, _, frame_indices, chunk, reference, ca_indices in tqdm(
        iterator, desc="Loading aligned reference"
    ):
        if reference_ca is None:
            reference_ca = reference.xyz[0, ca_indices, :].copy()
        ca_blocks.append(chunk.xyz[:, ca_indices, :].astype(np.float32))
        source_traj.extend([trajectory_index] * len(chunk))
        source_frame.extend(frame_indices.tolist())
    if not ca_blocks or reference_ca is None:
        raise RuntimeError("No reference frames were loaded")

    ca = np.concatenate(ca_blocks, axis=0)
    flat = ca.reshape(len(ca), -1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    raw_scores = pca.fit_transform(flat)
    score_mean = raw_scores.mean(axis=0)
    score_scale = raw_scores.std(axis=0, ddof=1)
    score_scale[score_scale == 0] = 1.0
    scores = (raw_scores - score_mean) / score_scale

    model = PCACV(
        reference_ca_nm=reference_ca,
        mean_flat=pca.mean_,
        components=pca.components_,
        score_mean=score_mean,
        score_scale=score_scale,
    )
    model_path = model.save(output_dir / "pca_cv.npz", pca.explained_variance_ratio_)
    np.savez_compressed(
        output_dir / "reference_cv.npz",
        cv=scores,
        source_trajectory=np.asarray(source_traj, dtype=int),
        source_frame=np.asarray(source_frame, dtype=int),
        trajectory_paths=np.asarray([str(path) for path in trajectories]),
    )

    # Save a small set of protein-only seed structures spanning PC1.
    seed_dir = output_dir / "seed_frames"
    seed_dir.mkdir(exist_ok=True)
    quantiles = np.linspace(0.02, 0.98, n_seed_frames)
    target_values = np.quantile(scores[:, 0], quantiles)
    chosen = [int(np.argmin(np.abs(scores[:, 0] - target))) for target in target_values]
    manifest_rows = []
    import mdtraj as md

    topology_frame = md.load(str(topology_pdb))
    protein_indices = topology_frame.topology.select(protein_selection)
    for seed_index, (quantile, sample_index) in enumerate(zip(quantiles, chosen)):
        traj_index = int(source_traj[sample_index])
        frame_index = int(source_frame[sample_index])
        frame = md.load_frame(
            str(trajectories[traj_index]),
            frame_index,
            top=str(topology_pdb),
            atom_indices=protein_indices,
        )
        seed_path = seed_dir / f"seed_{seed_index:03d}.pdb"
        frame.save_pdb(str(seed_path))
        manifest_rows.append(
            {
                "seed_index": seed_index,
                "quantile": float(quantile),
                "pc1": float(scores[sample_index, 0]),
                "source_trajectory": traj_index,
                "source_frame": frame_index,
                "pdb": str(seed_path),
            }
        )
    seed_fields = [
        "seed_index",
        "quantile",
        "pc1",
        "source_trajectory",
        "source_frame",
        "pdb",
    ]
    with (output_dir / "seed_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return model_path


def project_reference_trajectories(
    *,
    pca_model: str | Path,
    topology_pdb: str | Path,
    trajectories: list[str | Path],
    output_path: str | Path,
    protein_selection: str = "protein and chainid 0",
    stride: int = 1,
    burn_in_frames: int = 0,
    chunk_size: int = 1000,
    max_frames_per_trajectory: int | None = None,
) -> Path:
    """Project existing trajectories on a fitted PCA model at a matched time stride."""
    topology_pdb = Path(topology_pdb).resolve()
    trajectories = [Path(path).resolve() for path in trajectories]
    output_path = Path(output_path).resolve()
    model = PCACV.load(pca_model)
    score_blocks: list[np.ndarray] = []
    source_traj: list[int] = []
    source_frame: list[int] = []
    iterator = _iter_protein_chunks(
        topology_pdb,
        trajectories,
        protein_selection=protein_selection,
        stride=stride,
        burn_in_frames=burn_in_frames,
        chunk_size=chunk_size,
        max_frames_per_trajectory=max_frames_per_trajectory,
    )
    for trajectory_index, _, frame_indices, chunk, _, _ in tqdm(
        iterator, desc="Projecting matched reference"
    ):
        scores = model.project_trajectory(chunk)
        score_blocks.append(scores)
        source_traj.extend([trajectory_index] * len(scores))
        source_frame.extend(frame_indices.tolist())
    if not score_blocks:
        raise RuntimeError("No reference frames were projected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        cv=np.concatenate(score_blocks, axis=0),
        source_trajectory=np.asarray(source_traj, dtype=int),
        source_frame=np.asarray(source_frame, dtype=int),
        trajectory_paths=np.asarray([str(path) for path in trajectories]),
        stride=np.asarray(stride, dtype=int),
    )
    return output_path
