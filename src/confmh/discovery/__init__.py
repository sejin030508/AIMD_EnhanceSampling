"""MH-free controlled conformational exploration benchmarks.

This package intentionally does not import or call :mod:`confmh.mh`.  It evaluates
whether frozen autoregressive models can be steered toward useful conformations;
it does not claim equilibrium or kinetic correctness.
"""

from confmh.discovery.cv import ControlCV
from confmh.discovery.selection import select_candidate

__all__ = ["ControlCV", "select_candidate"]
