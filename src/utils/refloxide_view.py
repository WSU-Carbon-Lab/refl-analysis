"""Read-only pyref-shaped view over a fitted refloxide ``Objective``/``ReflectModel``.

Lets manuscript notebooks built around pyref's per-energy ``GlobalObjective``
API (``model.pol`` mutate-then-call, integer ``model.structure[i]`` indexing)
treat a NEW refloxide fit (``refloxide.model.ReflectModel`` /
``refloxide.objective.Objective``, e.g. as produced and pickled by
``refloxide/examples/real_data_repl.py``) as one more model to overlay --
WITHOUT reproducing pyref's full API. Nothing here needs to refit or sample
the refloxide result, only read its already-converged geometry and evaluate
its reflectivity, so there is no ``.setp``/``.pgen``/vary/constrain surface --
just enough to satisfy this repo's own ``rxr``/``model_reflectivity`` helpers
and structure-indexing code (e.g. ``fig_5_graded.ipynb``'s
``slab_film_profiles``).

Structural assumption (matches ``real_data_repl.py``'s own stack, and the
legacy pyref DFT/free stacks this repo already compares against): six slabs
in order ``vacuum | surface | bulk | interface | oxide | substrate``, so
``model.structure[i]`` lines up with the same indices the existing
pyref-oriented plotting helpers already assume.

Polarization labels: the refloxide fit pickle stores ``s``/``p`` opposite to
``read_xrr`` hub Data1D labels (``hub.s`` is identical to the objective
dataset's ``p`` column, and vice versa). When overlaying a refloxide model on
hub data, construct the view with ``swap_pol=True``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from refloxide.model import ReflectModel
    from refloxide.objective import Objective


class PolSwitchedModel:
    """Adapts refloxide ``ReflectModel`` to pyref ``model.pol = ...; model(q)``.

    refloxide's own ``ReflectModel.__call__(q, energy)`` always returns BOTH
    channels at once (``Reflectivity(s=..., p=...)``); pyref's ``ReflectModel``
    instead carries ``.pol`` as mutable state and returns one channel per
    call. This repo's own ``rxr``/``model_reflectivity`` helpers already
    mutate-then-restore ``.pol`` around a call -- this wrapper just gives them
    something with that exact shape to mutate, bound to one fixed energy.

    Parameters
    ----------
    model : refloxide.model.ReflectModel
        The fitted multi-energy model.
    energy : float
        Photon energy (eV) this view evaluates at.
    pol : {"s", "p"}, optional
        Starting polarization channel (hub / caller label).
    swap_pol : bool, optional
        When ``True``, map caller ``"s"`` to the refloxide ``p`` channel and
        caller ``"p"`` to the refloxide ``s`` channel so evaluations line up
        with ``read_xrr`` hub data.
    """

    def __init__(
        self,
        model: ReflectModel,
        energy: float,
        pol: Literal["s", "p"] = "s",
        *,
        swap_pol: bool = False,
    ) -> None:
        self._model = model
        self.energy = float(energy)
        self.pol = pol
        self.swap_pol = bool(swap_pol)

    @property
    def structure(self):
        """Shared structure; supports integer ``structure[i]`` indexing."""
        return self._model.structure

    def __call__(self, q: np.ndarray) -> np.ndarray:
        """Reflectivity at ``q`` for the CURRENTLY set ``self.pol`` channel."""
        result = self._model(np.asarray(q, dtype=float), self.energy)
        use_p = (self.pol == "p") ^ self.swap_pol
        return np.asarray(result.p if use_p else result.s, dtype=float)


class RefloxideObjectiveView:
    """One-energy pyref-shaped view over a refloxide ``Objective``.

    Parameters
    ----------
    objective : refloxide.objective.Objective
        The fitted multi-energy objective.
    energy : float
        Photon energy (eV) this view evaluates at.
    swap_pol : bool, optional
        Forwarded to :class:`PolSwitchedModel`. Use ``True`` when comparing
        against ``read_xrr`` hub polarization labels.
    """

    def __init__(
        self, objective: Objective, energy: float, *, swap_pol: bool = False
    ) -> None:
        self.objective = objective
        self.model = PolSwitchedModel(objective.model, energy, swap_pol=swap_pol)

    def varying_parameters(self):
        """Forwarded to the underlying ``Objective`` (shared across energies)."""
        return self.objective.varying_parameters()


def load_refloxide_fit(path: str | Path) -> Objective:
    """Load a refloxide ``Objective`` pickled by e.g. ``examples/real_data_repl.py``.

    Parameters
    ----------
    path : str or Path
        Pickle file path.

    Returns
    -------
    refloxide.objective.Objective
        Plain ``pickle.load`` -- refloxide's ``Objective`` is genuinely
        picklable (see ``Objective.__setstate__``/``thread_workers``'s own
        round-trip use of it), no pyref/refnx-legacy dependency needed to
        reload it.
    """
    with Path(path).open("rb") as f:
        return pickle.load(f)


def objective_view_at(
    objective: Objective, energy: float, *, swap_pol: bool = False
) -> RefloxideObjectiveView:
    """Single-energy pol-switchable view for ``rxr`` / ``slab_film_profiles``.

    Parameters
    ----------
    objective : refloxide.objective.Objective
    energy : float
        Photon energy (eV) to view.
    swap_pol : bool, optional
        When ``True``, remap polarization labels to match ``read_xrr`` hub data
        (required for manuscript overlays that plot hub ``.s``/``.p`` curves).

    Returns
    -------
    RefloxideObjectiveView
    """
    return RefloxideObjectiveView(objective, energy, swap_pol=swap_pol)


__all__ = [
    "PolSwitchedModel",
    "RefloxideObjectiveView",
    "load_refloxide_fit",
    "objective_view_at",
]
