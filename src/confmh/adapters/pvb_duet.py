from __future__ import annotations

"""PVB adapter for DuET-MD sampling.

The inner loop mirrors ``dyVAE.inference`` in ``yaledeus/PVB`` at commit
``c08e5e3c``, split so the bridge SDE can be paused at a checkpoint, branched,
and resumed with independent noise.  The preflight in
``scripts/duet/pvb_preflight`` verified that this split reproduces the official
loop to 1.1e-4 A, below the official loop's own 1.4e-4 A CUDA reproducibility
floor, so the vendor checkout is left unmodified.

Three things differ from the ConfRover adapter and are worth stating up front.

*Markov conditioning.*  PVB conditions one transition on the previous frame
alone, so ``prepare_history`` keeps only the last frame.  Nothing in the DuET
construction needs more: the history dependence of the target lives in the
program progress state, not in the emulator.

*Clean-frame prediction.*  PVB exposes no denoiser head, but its drift head is
trained against ``(x1 - xt) / (1 - t)``, so the endpoint is recovered exactly by
``x1_hat = xt + (1 - t) * drift``.  That is the model's own prediction, with the
same standing as ConfRover's ``pred_atom14``, not an externally defined forecast.

*Checkpoint granularity.*  The published inference config uses ten SDE steps, on
which 0.25 and 0.75 do not land.  Twenty steps put all four timings on exact
boundaries (5, 10, 15 and 18 of 20), so that is the default here; the official
ten stays available for reproducing the published baseline.  Either way the
realised progress is recorded on the particle state and reported by
``checkpoint_schedule`` rather than being silently rounded away.

``sde_step`` is the resolution of one bridge, not a trajectory length: the
physical rollout length is the outer horizon, and one PVB transition carries the
model's own 100 ps ATLAS lag regardless of how finely the bridge is discretised.
Matching a horizon against ConfRover therefore means matching the number of
autoregressive transitions, not the nominal physical time.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from confmh.adapters.base_iterative_frame import IterativeFrameAdapter
from confmh.adapters.confrover_duet import ConfRoverFrame
from confmh.duet.observables import ATOM37_INDEX
from confmh.utils import add_repo_to_path


@dataclass
class PVBHistoryState:
    """PVB is Markov: only the most recent frame conditions the next one."""

    coordinates_a: np.ndarray
    frame_count: int


@dataclass
class PVBParticleState:
    x_rep: Any            # encoder sample; constant through one transition
    xt: Any               # current bridge coordinate
    step: int             # index into the SDE grid
    count: int            # number of inner particles packed into the batch
    noise_seeds: np.ndarray
    cached_drift: Any = None
    cached_velocity: Any = None
    cached_guidance_grad: Any = None
    cached_guidance_log_phi: Any = None
    final_xt: Any = None
    realised_progress: list[float] = field(default_factory=list)


class PVBDuETAdapter(IterativeFrameAdapter):
    def __init__(
        self,
        *,
        repository_path: str | Path,
        checkpoint: str | Path,
        initial_structure: str | Path,
        device: str = "cuda:0",
        # 20 puts 0.25/0.50/0.75/0.90 on exact step boundaries; set 10 to
        # reproduce the published baseline discretisation.
        sde_step: int = 20,
        ca_adjacent_quality_threshold_a: float = 4.5,
        ca_adjacent_hard_threshold_a: float = 5.5,
        ca_adjacent_hard_tolerance_a: float = 1.0e-3,
        peptide_cn_min_a: float = 1.0,
        peptide_cn_max_a: float = 1.7,
        enforce_peptide_bond: bool = True,
    ) -> None:
        super().__init__()
        self.repository_path = Path(repository_path).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.initial_structure = Path(initial_structure).expanduser().resolve()
        self.device_name = str(device)
        self.sde_step = int(sde_step)
        if self.sde_step < 2:
            raise ValueError("sde_step must be at least 2")
        self.ca_adjacent_quality_threshold_a = float(ca_adjacent_quality_threshold_a)
        self.ca_adjacent_hard_threshold_a = float(ca_adjacent_hard_threshold_a)
        self.ca_adjacent_hard_tolerance_a = float(ca_adjacent_hard_tolerance_a)
        self.peptide_cn_min_a = float(peptide_cn_min_a)
        self.peptide_cn_max_a = float(peptide_cn_max_a)
        # Bundle A found broken peptide bonds in every ConfRover frame while the
        # C-alpha-only gate passed them, so this adapter checks the bond too.
        self.enforce_peptide_bond = bool(enforce_peptide_bond)

        self.model: Any = None
        self.topology: Any = None
        self._atom_index: np.ndarray | None = None
        self._atom37_slot: np.ndarray | None = None
        self._residue_of_atom: np.ndarray | None = None
        self._aatype: np.ndarray | None = None
        self._residue_count: int = 0
        self._batches: dict[int, dict[str, Any]] = {}
        self._initial_frame: ConfRoverFrame | None = None
        self._peptide_pairs: list[tuple[int, int]] = []
        self._guidance_potential: Any = None
        self._guidance_strength: float = 0.0
        self._guidance_update_steps: frozenset[int] = frozenset()

    # ------------------------------------------------------------------ setup
    def load_model(self) -> None:
        if self.model is not None:
            return
        for path in (self.repository_path, self.checkpoint, self.initial_structure):
            if not path.exists():
                raise FileNotFoundError(path)
        add_repo_to_path(self.repository_path)

        import os

        os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")
        import mdtraj as md
        import torch

        self.model = torch.load(
            str(self.checkpoint), map_location="cpu", weights_only=False
        )
        self.model.to(self.device_name)
        self.model.eval()
        # Guidance differentiates through the frozen decoder with respect to
        # coordinates only.  Keeping parameters frozen avoids allocating their
        # gradients and cannot change the baseline forward pass.
        self.model.requires_grad_(False)
        torch.set_grad_enabled(False)

        from data import make_batch

        _, (atom_index,) = make_batch(str(self.initial_structure), 1)
        self._atom_index = np.asarray(atom_index, dtype=int)
        state0 = md.load(str(self.initial_structure))
        kept = state0.atom_slice(self._atom_index)
        self.topology = kept.topology
        self._build_atom37_mapping()
        # mdtraj carries nanometres; the model and atom37 both use Angstrom.
        self._initial_frame = self.frame_from_coordinates(
            np.asarray(kept.xyz[0], dtype=float) * 10.0
        )

    def configure_guidance(
        self,
        potential: Any,
        *,
        strength: float,
        update_steps: Sequence[int] | None = None,
    ) -> None:
        """Attach a frozen differentiable endpoint potential to the bridge.

        ``update_steps`` contains zero-based PVB update indices.  The final
        deterministic update is intentionally excluded: it has no Gaussian
        base/proposal density ratio and therefore cannot be shifted while
        retaining the exact importance correction used here.
        """
        self.load_model()
        eta = float(strength)
        if not np.isfinite(eta) or eta < 0.0:
            raise ValueError("guidance strength must be finite and non-negative")
        if int(getattr(potential, "atom_count", -1)) != int(self.topology.n_atoms):
            raise ValueError("Guidance potential atom count does not match PVB")
        if update_steps is None:
            steps = frozenset(range(self.sde_step - 1))
        else:
            steps = frozenset(int(item) for item in update_steps)
            invalid = sorted(
                item for item in steps if item < 0 or item >= self.sde_step - 1
            )
            if invalid:
                raise ValueError(
                    "Guidance can only target stochastic PVB updates; invalid "
                    f"indices: {invalid}"
                )
        self._guidance_potential = potential.to(self.device_name)
        self._guidance_potential.eval()
        self._guidance_potential.requires_grad_(False)
        self._guidance_strength = eta
        self._guidance_update_steps = steps

    @property
    def guidance_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self._guidance_potential is not None,
            "strength": float(self._guidance_strength),
            "update_steps": sorted(self._guidance_update_steps),
            "stochastic_update_count": self.sde_step - 1,
            "final_deterministic_update_guided": False,
        }

    def _build_atom37_mapping(self) -> None:
        """Map each retained PVB atom onto its atom37 slot.

        atom37 is a heavy-atom layout, and PVB drops hydrogens, so the two line
        up on name.  Any atom without an atom37 slot is dropped from the frame
        rather than written to an arbitrary index.
        """
        from confmh.duet.observables import ATOM37_NAMES  # noqa: F401

        residues = list(self.topology.residues)
        self._residue_count = len(residues)
        residue_lookup = {residue.index: order for order, residue in enumerate(residues)}
        slots, residue_ids = [], []
        unmapped: set[str] = set()
        for atom in self.topology.atoms:
            slot = ATOM37_INDEX.get(atom.name)
            if slot is None:
                unmapped.add(atom.name)
                slots.append(-1)
            else:
                slots.append(slot)
            residue_ids.append(residue_lookup[atom.residue.index])
        self._atom37_slot = np.asarray(slots, dtype=int)
        self._residue_of_atom = np.asarray(residue_ids, dtype=int)
        self.unmapped_atom_names = sorted(unmapped)

        from confmh.duet.observables import ATOM37_NAMES as NAMES  # noqa: F401

        three_to_index = {}
        try:
            from openfold.np import residue_constants as rc

            three_to_index = {
                name: rc.restype_order.get(rc.restype_3to1.get(name, "X"), 20)
                for name in {residue.name for residue in residues}
            }
        except Exception:
            three_to_index = {}
        self._aatype = np.asarray(
            [three_to_index.get(residue.name, 20) for residue in residues], dtype=int
        )

        nitrogen = ATOM37_INDEX["N"]
        carbon = ATOM37_INDEX["C"]
        self._peptide_pairs = [
            (index, index + 1) for index in range(self._residue_count - 1)
        ]
        self._peptide_slots = (carbon, nitrogen)

    def checkpoint_schedule(self, progresses: Sequence[float]) -> list[dict[str, Any]]:
        """Report where each requested progress actually lands on the SDE grid."""
        rows = []
        for progress in progresses:
            step = int(round(self.sde_step * float(progress)))
            step = max(1, min(step, self.sde_step - 1))
            rows.append(
                {
                    "requested_progress": float(progress),
                    "step": step,
                    "realised_progress": step / self.sde_step,
                    "exact": abs(step / self.sde_step - float(progress)) < 1e-12,
                }
            )
        return rows

    # ------------------------------------------------------------------ frames
    def frame_from_coordinates(self, coordinates_a: np.ndarray) -> ConfRoverFrame:
        """Pack PVB's heavy-atom coordinates into the atom37 layout."""
        assert self._atom37_slot is not None and self._residue_of_atom is not None
        atom37 = np.zeros((self._residue_count, 37, 3), dtype=np.float32)
        mask = np.zeros((self._residue_count, 37), dtype=bool)
        usable = self._atom37_slot >= 0
        atom37[self._residue_of_atom[usable], self._atom37_slot[usable]] = (
            coordinates_a[usable]
        )
        mask[self._residue_of_atom[usable], self._atom37_slot[usable]] = True
        return ConfRoverFrame(
            atom37_a=atom37, atom37_mask=mask, aatype=np.asarray(self._aatype)
        )

    def initial_history(self) -> list[ConfRoverFrame]:
        self.load_model()
        assert self._initial_frame is not None
        return [self._initial_frame]

    def _coordinates_from_frame(self, frame: Any) -> np.ndarray:
        assert self._atom37_slot is not None and self._residue_of_atom is not None
        atom37 = np.asarray(frame.atom37_a, dtype=float)
        usable = self._atom37_slot >= 0
        coordinates = np.zeros((len(self._atom37_slot), 3), dtype=float)
        coordinates[usable] = atom37[
            self._residue_of_atom[usable], self._atom37_slot[usable]
        ]
        return coordinates

    # ------------------------------------------------------------------ batch
    def _batch_for(self, count: int) -> dict[str, Any]:
        if count not in self._batches:
            from data import make_batch

            batch, _ = make_batch(str(self.initial_structure), int(count))
            self._batches[count] = {
                key: (value.to(self.device_name) if hasattr(value, "to") else value)
                for key, value in batch.items()
            }
        return self._batches[count]

    # --------------------------------------------------------------- interface
    def prepare_history(self, history: Sequence[Any]) -> PVBHistoryState:
        self.load_model()
        if not history:
            raise ValueError("history must contain at least one frame")
        # PVB is Markov; only the last frame conditions the next transition.
        return PVBHistoryState(
            coordinates_a=self._coordinates_from_frame(history[-1]),
            frame_count=len(history),
        )

    def initialize_inner_particles(
        self, history_state: PVBHistoryState, count: int, seeds: Sequence[int]
    ) -> PVBParticleState:
        self.load_model()
        import torch
        from module.graph import construct_edges

        if len(seeds) != count:
            raise ValueError("Expected one deterministic seed per inner particle")
        batch = self._batch_for(count)
        per_particle = torch.as_tensor(
            history_state.coordinates_a, dtype=torch.float32, device=self.device_name
        )
        x = per_particle.repeat(count, 1).contiguous()
        z, b = batch["atype"], batch["btype"]
        abid, edge_mask, bond_index = (
            batch["abid"], batch["edge_mask"], batch["bond_index"],
        )
        e_index, e_weight, e_vec, bond_type = construct_edges(
            Z=z, X=x, bid=abid, mask=edge_mask, bond_index=bond_index,
            cutoff_lower=self.model.cutoff_lower,
            cutoff_upper=self.model.cutoff_upper,
            cutoff_H=self.model.cutoff_H,
            k_neighbors=self.model.k_neighbors,
        )
        # The encoder is stochastic (x_rep = x + noise), so the particles differ
        # from step 0 and x_rep must travel with xt through every resampling.
        # It draws one tensor over the whole packed batch, so the family is
        # seeded as a family rather than per particle; that is fine because the
        # draw happens once, before any branching, and every later divergence
        # comes from the per-particle SDE noise below.
        devices = (
            [torch.device(self.device_name).index or 0]
            if self.device_name.startswith("cuda")
            else []
        )
        family_seed = int(
            np.random.SeedSequence([int(seed) for seed in seeds]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(family_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(family_seed)
            x_rep, _ = self.model.encode(
                z, b, x, x, abid, e_index, e_weight, e_vec, bond_type,
                torch.ones_like(z).bool(),
            )
        return PVBParticleState(
            x_rep=x_rep,
            xt=x_rep.clone(),
            step=0,
            count=int(count),
            noise_seeds=np.asarray(seeds, dtype=np.int64),
        )

    def _grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0 - 1.0 / self.sde_step, self.sde_step)

    def _decode(self, state: PVBParticleState) -> None:
        import torch
        from module.graph import construct_edges

        batch = self._batch_for(state.count)
        z, b = batch["atype"], batch["btype"]
        abid, edge_mask, bond_index = (
            batch["abid"], batch["edge_mask"], batch["bond_index"],
        )
        grid = self._grid()
        t = float(grid[state.step])
        ones = torch.ones(
            state.xt.shape[0], 1, dtype=state.xt.dtype, device=state.xt.device
        )
        d_index, d_weight_t, d_vec_t, bond_type = construct_edges(
            Z=z, X=state.xt, bid=abid, mask=edge_mask, bond_index=bond_index,
            cutoff_lower=self.model.cutoff_lower,
            cutoff_upper=self.model.cutoff_upper,
            cutoff_H=self.model.cutoff_H,
            k_neighbors=self.model.k_neighbors,
        )
        atoms_0 = state.x_rep[d_index.transpose(0, 1)]
        d_vec_0 = atoms_0[:, 0] - atoms_0[:, 1]
        d_weight_0 = torch.norm(d_vec_0, dim=-1)
        velocity, drift = self.model.decode(
            z, b, state.xt, ones * t, abid, d_index, d_weight_0, d_vec_0,
            d_weight_t, d_vec_t, bond_type,
        )
        state.cached_velocity = velocity
        state.cached_drift = drift
        self.accounting.reverse_decoder_evaluations += state.count

    def _predicted_endpoint_tensor(self, state: PVBParticleState):
        if state.cached_drift is None:
            self._decode(state)
        t = float(self._grid()[state.step])
        return state.xt + (1.0 - t) * state.cached_drift

    def _guided_decode(self, state: PVBParticleState) -> None:
        """Decode once and cache both endpoint score and its coordinate gradient."""
        if self._guidance_potential is None:
            raise RuntimeError("configure_guidance() must be called first")
        import torch
        from module.graph import construct_edges

        should_differentiate = bool(
            self._guidance_strength > 0.0
            and state.step in self._guidance_update_steps
            and state.step < self.sde_step - 1
        )
        if not should_differentiate:
            # This branch deliberately calls the unchanged baseline decoder so
            # eta=0 is bitwise-identical to ordinary PVB given identical seeds.
            if state.cached_drift is None:
                self._decode(state)
            with torch.no_grad():
                predicted = self._predicted_endpoint_tensor(state)
                state.cached_guidance_log_phi = self._guidance_potential.forward_flat(
                    predicted, state.count
                ).detach()
            self.accounting.predicted_clean_evaluations += state.count
            self.accounting.guidance_potential_evaluations += state.count
            state.cached_guidance_grad = None
            return

        batch = self._batch_for(state.count)
        z, b = batch["atype"], batch["btype"]
        abid, edge_mask, bond_index = (
            batch["abid"], batch["edge_mask"], batch["bond_index"],
        )
        t = float(self._grid()[state.step])
        with torch.enable_grad():
            current = state.xt.detach().requires_grad_(True)
            ones = torch.ones(
                current.shape[0], 1, dtype=current.dtype, device=current.device
            )
            d_index, d_weight_t, d_vec_t, bond_type = construct_edges(
                Z=z, X=current, bid=abid, mask=edge_mask, bond_index=bond_index,
                cutoff_lower=self.model.cutoff_lower,
                cutoff_upper=self.model.cutoff_upper,
                cutoff_H=self.model.cutoff_H,
                k_neighbors=self.model.k_neighbors,
            )
            x_rep = state.x_rep.detach()
            atoms_0 = x_rep[d_index.transpose(0, 1)]
            d_vec_0 = atoms_0[:, 0] - atoms_0[:, 1]
            d_weight_0 = torch.norm(d_vec_0, dim=-1)
            velocity, drift = self.model.decode(
                z, b, current, ones * t, abid, d_index, d_weight_0, d_vec_0,
                d_weight_t, d_vec_t, bond_type,
            )
            predicted = current + (1.0 - t) * drift
            log_phi = self._guidance_potential.forward_flat(predicted, state.count)
            gradient = torch.autograd.grad(log_phi.sum(), current, create_graph=False)[0]
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Non-finite PVB guidance gradient")
        state.cached_velocity = velocity.detach()
        state.cached_drift = drift.detach()
        state.cached_guidance_grad = gradient.detach()
        state.cached_guidance_log_phi = log_phi.detach()
        self.accounting.reverse_decoder_evaluations += state.count
        self.accounting.predicted_clean_evaluations += state.count
        self.accounting.guidance_backward_evaluations += state.count
        self.accounting.guidance_potential_evaluations += state.count

    @staticmethod
    def _gaussian_shift_log_ratio(
        noise: Any, mean_shift: Any, standard_deviation: Any, count: int
    ):
        """Return per-candidate log p_base/q_guided for a guided draw.

        The actual draw is ``guided_mean + std * noise``.  Consequently the
        standardized residual under the base kernel is ``noise + shift/std``.
        This yields ``-noise.dot(shift/std) - 0.5*||shift/std||^2``.
        """
        scaled = mean_shift / standard_deviation
        atoms = noise.shape[0] // int(count)
        return (
            -(noise * scaled).reshape(count, atoms * 3).sum(-1)
            - 0.5 * scaled.square().reshape(count, atoms * 3).sum(-1)
        )

    def _guided_integrate(self, state: PVBParticleState):
        """Take one base/guided update and return its exact proposal correction."""
        import torch

        should_guide = bool(
            self._guidance_strength > 0.0
            and state.step in self._guidance_update_steps
            and state.step < self.sde_step - 1
        )
        if state.cached_drift is None:
            if should_guide:
                self._guided_decode(state)
            else:
                self._decode(state)
        dt = float(self._grid()[1] - self._grid()[0])
        drift = state.cached_drift
        correction = torch.zeros(
            state.count, dtype=state.xt.dtype, device=state.xt.device
        )
        if state.step == self.sde_step - 1:
            # The published PVB loop ends in a deterministic update.  It stays
            # unmodified because no Gaussian density correction exists here.
            state.xt = state.xt + drift * dt
        else:
            noise = self._particle_noise(state)
            sigma = torch.as_tensor(
                self.model.sigma, dtype=state.xt.dtype, device=state.xt.device
            )
            standard_deviation = sigma * np.sqrt(dt)
            if state.cached_guidance_grad is not None:
                mean_shift = (
                    self._guidance_strength * sigma.square()
                    * state.cached_guidance_grad * dt
                )
                correction = self._gaussian_shift_log_ratio(
                    noise, mean_shift, standard_deviation, state.count
                )
                state.xt = (
                    state.xt + drift * dt + mean_shift
                    + standard_deviation * noise
                )
                self.accounting.guidance_log_ratio_evaluations += state.count
            else:
                state.xt = state.xt + drift * dt + standard_deviation * noise
        if not bool(torch.isfinite(state.xt).all()) or not bool(
            torch.isfinite(correction).all()
        ):
            raise FloatingPointError("Non-finite guided PVB update")
        state.final_xt = state.xt
        state.cached_drift = None
        state.cached_velocity = None
        state.cached_guidance_grad = None
        state.cached_guidance_log_phi = None
        state.step += 1
        # Keep corrections on device across a bridge segment.  The public
        # checkpoint methods synchronize once, rather than once per SDE step.
        return correction.detach().to(torch.float64)

    def _integrate(self, state: PVBParticleState) -> None:
        """Take one Euler-Maruyama step using the cached drift."""
        if state.cached_drift is None:
            self._decode(state)
        grid = self._grid()
        dt = float(grid[1] - grid[0])
        drift = state.cached_drift
        if state.step == self.sde_step - 1:
            state.xt = state.xt + drift * dt
        else:
            noise = self._particle_noise(state)
            state.xt = state.xt + drift * dt + self.model.sigma * np.sqrt(dt) * noise
        state.final_xt = state.xt
        state.cached_drift = None
        state.cached_velocity = None
        state.step += 1

    def _particle_noise(self, state: PVBParticleState):
        """Per-particle deterministic noise.

        The batch packs every particle into one atom dimension, so drawing one
        tensor with a global RNG would tie a particle's noise to its position in
        the batch.  Seeding per particle instead keeps a resampled child's
        stream tied to its own seed, which is what reseeding after a branch
        depends on.
        """
        import torch

        atoms = state.xt.shape[0] // state.count
        blocks = []
        for index in range(state.count):
            seed = int(
                np.random.SeedSequence(
                    [int(state.noise_seeds[index]), 1, int(state.step)]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            blocks.append(
                torch.randn(
                    (atoms, 3), generator=generator, dtype=state.xt.dtype
                ).to(state.xt.device)
            )
        return torch.cat(blocks, dim=0)

    def denoise_to_checkpoint(
        self, particle_state: PVBParticleState, checkpoint_progress: float
    ) -> PVBParticleState:
        target = int(round(self.sde_step * float(checkpoint_progress)))
        target = max(1, min(target, self.sde_step - 1))
        while particle_state.step < target:
            self._integrate(particle_state)
        if particle_state.cached_drift is None:
            self._decode(particle_state)
        particle_state.realised_progress.append(target / self.sde_step)
        return particle_state

    def predict_clean(self, particle_state: PVBParticleState) -> list[Any]:
        if particle_state.cached_drift is None:
            self._decode(particle_state)
        # Drift head target is (x1 - xt) / (1 - t); inverting it gives the
        # model's own endpoint estimate.
        predicted = self._predicted_endpoint_tensor(particle_state)
        self.accounting.predicted_clean_evaluations += particle_state.count
        return self._split_frames(predicted, particle_state.count)

    def _split_frames(self, tensor: Any, count: int) -> list[ConfRoverFrame]:
        array = tensor.detach().float().cpu().numpy()
        atoms = array.shape[0] // count
        return [
            self.frame_from_coordinates(array[index * atoms:(index + 1) * atoms])
            for index in range(count)
        ]

    def resample_particle_state(
        self, particle_state: PVBParticleState, ancestor_indices: Any
    ) -> PVBParticleState:
        import torch

        ancestors = np.asarray(ancestor_indices, dtype=int)
        count = particle_state.count
        atoms = particle_state.xt.shape[0] // count
        order = torch.as_tensor(
            np.concatenate(
                [np.arange(a * atoms, (a + 1) * atoms) for a in ancestors]
            ),
            dtype=torch.long,
            device=particle_state.xt.device,
        )
        gathered = PVBParticleState(
            # x_rep conditions every decode, so it must follow its child.
            x_rep=particle_state.x_rep.index_select(0, order).clone(),
            xt=particle_state.xt.index_select(0, order).clone(),
            step=particle_state.step,
            count=count,
            noise_seeds=particle_state.noise_seeds[ancestors].copy(),
            realised_progress=list(particle_state.realised_progress),
        )
        for name in ("cached_drift", "cached_velocity", "cached_guidance_grad"):
            value = getattr(particle_state, name)
            if value is not None:
                setattr(gathered, name, value.index_select(0, order).clone())
        if particle_state.cached_guidance_log_phi is not None:
            selected = torch.as_tensor(
                ancestors, dtype=torch.long,
                device=particle_state.cached_guidance_log_phi.device,
            )
            gathered.cached_guidance_log_phi = (
                particle_state.cached_guidance_log_phi.index_select(0, selected).clone()
            )
        return gathered

    def reseed_particle_state(
        self, particle_state: PVBParticleState, independent_seeds: Sequence[int]
    ) -> PVBParticleState:
        if len(independent_seeds) != particle_state.count:
            raise ValueError("Expected one independent continuation seed per particle")
        particle_state.noise_seeds = np.asarray(independent_seeds, dtype=np.int64)
        return particle_state

    def denoise_to_end(
        self, particle_state: PVBParticleState, independent_seeds: Sequence[int]
    ) -> PVBParticleState:
        particle_state = self.reseed_particle_state(particle_state, independent_seeds)
        while particle_state.step < self.sde_step:
            self._integrate(particle_state)
        return particle_state

    # ----------------------------------------------------------- guided PVB
    def guided_denoise_to_checkpoint(
        self, particle_state: PVBParticleState, checkpoint_progress: float
    ) -> tuple[PVBParticleState, np.ndarray]:
        target = int(round(self.sde_step * float(checkpoint_progress)))
        target = max(1, min(target, self.sde_step - 1))
        correction = None
        while particle_state.step < target:
            step_correction = self._guided_integrate(particle_state)
            correction = (
                step_correction if correction is None
                else correction + step_correction
            )
        if particle_state.cached_guidance_log_phi is None:
            self._guided_decode(particle_state)
        particle_state.realised_progress.append(target / self.sde_step)
        if correction is None:
            correction_array = np.zeros(particle_state.count, dtype=np.float64)
        else:
            correction_array = correction.cpu().numpy()
        return particle_state, correction_array

    def guided_checkpoint_log_potential(
        self, particle_state: PVBParticleState
    ) -> np.ndarray:
        import torch

        if particle_state.cached_guidance_log_phi is None:
            self._guided_decode(particle_state)
        return (
            particle_state.cached_guidance_log_phi.detach()
            .to("cpu", dtype=torch.float64)
            .numpy()
        )

    def guided_denoise_to_end(
        self,
        particle_state: PVBParticleState,
        independent_seeds: Sequence[int] | None,
    ) -> tuple[PVBParticleState, np.ndarray]:
        if independent_seeds is not None:
            particle_state = self.reseed_particle_state(
                particle_state, independent_seeds
            )
        correction = None
        while particle_state.step < self.sde_step:
            step_correction = self._guided_integrate(particle_state)
            correction = (
                step_correction if correction is None
                else correction + step_correction
            )
        if correction is None:
            correction_array = np.zeros(particle_state.count, dtype=np.float64)
        else:
            correction_array = correction.cpu().numpy()
        return particle_state, correction_array

    def guided_endpoint_log_potential(
        self, particle_state: PVBParticleState
    ) -> np.ndarray:
        if self._guidance_potential is None or particle_state.final_xt is None:
            raise RuntimeError("Guided PVB bridge has not been configured/completed")
        import torch

        with torch.no_grad():
            values = self._guidance_potential.forward_flat(
                particle_state.final_xt, particle_state.count
            )
        self.accounting.guidance_potential_evaluations += particle_state.count
        return values.detach().to("cpu", dtype=torch.float64).numpy()

    def finalize_frames(self, particle_state: PVBParticleState) -> list[Any]:
        if particle_state.final_xt is None:
            raise RuntimeError("PVB particles did not complete the bridge")
        frames = self._split_frames(particle_state.final_xt, particle_state.count)
        self.accounting.generated_complete_frames += len(frames)
        return frames

    # ------------------------------------------------------------- validation
    def validate_frames(self, frames: Sequence[Any]) -> list[dict[str, Any]]:
        carbon, nitrogen = self._peptide_slots
        ca = ATOM37_INDEX["CA"]
        results = []
        for frame in frames:
            atom37 = np.asarray(frame.atom37_a, dtype=float)
            mask = np.asarray(frame.atom37_mask, dtype=bool)
            ca_a = atom37[:, ca, :]
            adjacent = np.linalg.norm(np.diff(ca_a, axis=0), axis=-1)
            quality = adjacent >= self.ca_adjacent_quality_threshold_a
            nonfinite = int(np.size(atom37) - np.isfinite(atom37).sum())
            pairwise = np.linalg.norm(ca_a[:, None, :] - ca_a[None, :, :], axis=-1)
            nonneighbor = np.triu(np.ones(pairwise.shape, dtype=bool), k=2)
            clashes = int(np.sum((pairwise < 1.0) & nonneighbor))

            lengths = []
            for left, right in self._peptide_pairs:
                if mask[left, carbon] and mask[right, nitrogen]:
                    lengths.append(
                        float(
                            np.linalg.norm(
                                atom37[left, carbon] - atom37[right, nitrogen]
                            )
                        )
                    )
            peptide = np.asarray(lengths, dtype=float)
            peptide_bad = (
                int(
                    np.sum(
                        (peptide < self.peptide_cn_min_a)
                        | (peptide > self.peptide_cn_max_a)
                    )
                )
                if len(peptide)
                else 0
            )
            valid = bool(
                nonfinite == 0
                and clashes == 0
                and np.all(
                    adjacent
                    <= self.ca_adjacent_hard_threshold_a
                    + self.ca_adjacent_hard_tolerance_a
                )
                and (not self.enforce_peptide_bond or peptide_bad == 0)
            )
            results.append(
                {
                    "valid": valid,
                    "nonfinite_coordinate_count": nonfinite,
                    "ca_adjacent_max_a": float(np.max(adjacent)) if len(adjacent) else 0.0,
                    "ca_adjacent_count_ge_4_5a": int(np.sum(quality)),
                    "ca_adjacent_fraction_ge_4_5a": (
                        float(np.mean(quality)) if len(adjacent) else 0.0
                    ),
                    "ca_adjacent_quality_threshold_a": self.ca_adjacent_quality_threshold_a,
                    "ca_adjacent_hard_threshold_a": self.ca_adjacent_hard_threshold_a,
                    "ca_adjacent_hard_tolerance_a": self.ca_adjacent_hard_tolerance_a,
                    "ca_clash_count_lt_1a": clashes,
                    "peptide_cn_min_a": float(peptide.min()) if len(peptide) else None,
                    "peptide_cn_max_a": float(peptide.max()) if len(peptide) else None,
                    "peptide_cn_violation_count": peptide_bad,
                    "peptide_cn_enforced": self.enforce_peptide_bond,
                }
            )
        return results
