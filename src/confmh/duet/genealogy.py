from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def surviving_ancestor_count(lineages: Sequence[Sequence[int]], generation: int = 0) -> int:
    ancestors = [lineage[generation] for lineage in lineages if len(lineage) > generation]
    return len(set(ancestors))


def genealogical_entropy(lineages: Sequence[Sequence[int]], generation: int = 0) -> float:
    ancestors = [lineage[generation] for lineage in lineages if len(lineage) > generation]
    if not ancestors:
        return 0.0
    counts = np.asarray(list(Counter(ancestors).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))

