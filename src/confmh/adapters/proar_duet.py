from __future__ import annotations


class ProARDuETAdapter:
    """Optional transfer adapter placeholder guarded by the Phase-F gate.

    ProAR integration is intentionally not activated until the ConfRover result
    supports the two-clock hypothesis and a stable stochastic intermediate
    refinement state is confirmed.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Phase F is gated: first establish the ConfRover two-clock result and "
            "confirm a stochastic ProAR refinement checkpoint."
        )

