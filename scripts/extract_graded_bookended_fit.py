"""One-time extraction: pull a portable summary out of the legacy graded
book-ended pyref ``AnisotropyObjective`` pickle at
``@models/xrr/znpc/graded/graded_fit.pkl``, so refloxide's
``examples/bookended_nonres_repl.py`` can rebuild the same
``BookendedOrientationProfile`` geometry without needing pyref/refnx-legacy
installed.

Run once, inside this repo's own venv (has pyref + refnx):

    cd ~/projects/refl-analysis && uv run python scripts/extract_graded_bookended_fit.py

Unlike the DFT/free-tensor extractions, this pickle is a single-energy
``AnisotropyObjective`` (283.7 eV, ``pol='sp'``), not a ``GlobalObjective`` --
confirmed directly by loading it. Its structure is ``Vacuum | ZnPc
(BookendedOrientationProfile) | Oxide (SiO2) | Substrate (Si)``, the same
topology ``refloxide/examples/bookended_repl.py`` already builds with
refloxide's ``BookendedOrientationProfile``/``BookendedComponent``.

Only the film's own (thick/rough/density/tau/alpha) shape parameters and the
surrounding slabs' (thick, rough, density, formula) are extracted -- the OOC
table itself is NOT re-embedded, since it's the same
``@models/optical/znpc/dft.csv`` every other refloxide ZnPc example already
reads directly (confirmed identical by comparing ``anchor.values_at(...)``
against that file's tabulated points).

Writes, next to the source pickle:

- ``graded_fit_summary.json`` -- film profile parameters, surrounding slab
  geometry, model instrumentation, and the source OOC csv path.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

PICKLE_PATH = Path.home() / "projects/refl-analysis/@models/xrr/znpc/graded/graded_fit.pkl"
OUT_DIR = PICKLE_PATH.parent
SUMMARY_PATH = OUT_DIR / "graded_fit_summary.json"
OOC_CSV = Path.home() / "projects/refl-analysis/@models/optical/znpc/dft.csv"

FILM_PARAM_NAMES = (
    "total_thick",
    "surface_roughness",
    "tau_si",
    "tau_vac",
    "alpha_bulk",
    "alpha_si",
    "alpha_vac",
    "density_bulk",
    "density_si",
    "density_vac",
)

print(f"loading {PICKLE_PATH} ...", flush=True)
with open(PICKLE_PATH, "rb") as f:
    objective = pickle.load(f)
print(f"loaded: {type(objective)}", flush=True)

model = objective.model
structure = model.structure

from refloxide.pxr.energy.bookended import BookendedOrientationProfile  # noqa: E402

film = next(c for c in structure.components if isinstance(c, BookendedOrientationProfile))
print(f"film: {film.name}, num_slabs={film.num_slabs}, mesh_constant={film.mesh_constant}")

film_params = {name: float(getattr(film, name).value or 0.0) for name in FILM_PARAM_NAMES}
for name, value in film_params.items():
    print(f"  {name}: {value}")

layers = []
for slab in structure.components:
    if slab is film:
        continue
    sld = slab.sld
    layer = {
        "name": slab.name.rsplit("_", 1)[0],
        "thick": float(slab.thick.value or 0.0),
        "rough": float(slab.rough.value or 0.0),
        "density": float(sld.density.value or 0.0),
        "formula": getattr(sld, "formula", ""),
    }
    layers.append(layer)
    print(f"  layer: {layer}", flush=True)

instrumentation = {
    "scale_s": float(model.scale_s.value or 0.0),
    "scale_p": float(model.scale_p.value or 0.0),
    "theta_offset_s": float(model.theta_offset_s.value or 0.0),
    "theta_offset_p": float(model.theta_offset_p.value or 0.0),
    "bkg": float(model.bkg.value or 0.0),
    "dq": float(model.dq.value or 0.0),
    "q_offset": float(model.q_offset.value or 0.0),
    "energy_offset": float(model.energy_offset.value or 0.0),
}
print(f"instrumentation: {instrumentation}", flush=True)

summary = {
    "source_pickle": str(PICKLE_PATH),
    "energy": float(model.energy),
    "ooc_csv": str(OOC_CSV),
    "film": {
        "num_slabs": int(film.num_slabs),
        "mesh_constant": float(film.mesh_constant),
        "params": film_params,
    },
    "layers": layers,
    "instrumentation": instrumentation,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
print(f"wrote {SUMMARY_PATH}", flush=True)
print("DONE", flush=True)
