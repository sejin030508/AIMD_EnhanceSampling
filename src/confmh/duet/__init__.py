"""DuET-MD: dual-time steering for frozen full-history dynamics emulators.

The package targets bias-conditioned transition-path candidates under a frozen
surrogate prior.  It does not implement exact dynamics under a modified
potential and must not be used for rate or MFPT estimation.
"""

from confmh.duet.programs import Event, ProgressState, TemporalProgram

__all__ = ["Event", "ProgressState", "TemporalProgram"]

