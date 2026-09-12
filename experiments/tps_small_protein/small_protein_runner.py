#!/usr/bin/env python3
"""Isolated entry point for the small-protein endpoint pilot.

The existing Phase-B pocket module intentionally rejects unknown protein names
when computing pocket-specific hidden observables.  Small proteins have no such
sampling/evaluation feature, so this wrapper replaces that one diagnostic with
an empty mapping only in this process.  No shared Phase-A/B source is edited.
"""

from __future__ import annotations

from typing import Any, Mapping

from confmh.duet import phase_b_pockets


_ORIGINAL_HIDDEN_OBSERVABLES = phase_b_pockets.hidden_observables
_SMALL_PROTEINS = {"chignolin", "trpcage", "bba"}


def _isolated_hidden_observables(
    frame: Any,
    *,
    protein: str,
    uniprot_to_model_index: Mapping[int, int],
    metric: Any,
) -> dict[str, float | None]:
    if protein in _SMALL_PROTEINS:
        return {}
    return _ORIGINAL_HIDDEN_OBSERVABLES(
        frame,
        protein=protein,
        uniprot_to_model_index=uniprot_to_model_index,
        metric=metric,
    )


phase_b_pockets.hidden_observables = _isolated_hidden_observables

from confmh.duet.phase_b_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
