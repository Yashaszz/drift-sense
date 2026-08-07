"""Drift-Sense: navigation-error recovery for wafer-inspection stage drift.

The package locates a zoomed-in SEM ``reference`` image inside a zoomed-out
``search`` image and reports the centre of the match in search-image pixels.

Module map
----------
``config``
    Every coordinate/scale convention constant, in one place.
``types``
    Shared dataclasses that cross module boundaries. The real interface contract.
``matcher``
    Stages 2, 3, 3b and 5 — template construction, ZNCC, peak extraction, subpixel.
``confidence``
    Stage 6 — calibrated confidence and the low-confidence flag.
``localize``
    The submitted deliverable. Orchestrates every stage; never raises.

Modules owned by other roles (``layouts``, ``render``, ``generate_dataset``,
``sem_physics``, ``pose``, ``disambiguate``, ``baseline_ncc``, ``evaluate``)
land alongside these and import from ``config`` and ``types``.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
