"""One-time extraction: pull a portable summary + data/predictions out of the
legacy pyref GlobalObjective pickle at
``@models/xrr/znpc/dft/dft_en_offset_new2.pkl``, so refloxide's comparison
example (``refloxide/examples/dft_model_comparison_repl.py``) can load it
without needing pyref/refnx-legacy installed.

Run once, inside this repo's own venv (has pyref + refnx):

    cd ~/projects/refl-analysis && uv run python scripts/extract_dft_globalobjective.py

Writes, next to the source pickle:

- ``dft_en_offset_new2_summary.json`` -- anchor-energy (283.7 eV) structure
  geometry (every other energy's slab parameters are refnx constraints
  pointing at this one, confirmed by inspecting the pickle directly) plus
  the per-energy instrument corrections.
- ``dft_en_offset_new2_data.parquet`` -- measured data and the legacy
  model's own predicted curves, per energy per NATIVE polarization label
  (already de-inverted from pyref's legacy pol='s'/'p' convention -- see
  the note below).

Label note, confirmed directly from ``pyref.fitting.model.ReflectModel``
source, not assumed:

- ``.model()``'s pol-selection: ``pol='s' -> refl[:, 1, 1]`` (R_pp, native
  "p"); ``pol='p' -> refl[:, 0, 0]`` (R_ss, native "s").
- ``reflectivity()``'s scale application is unconditional on ``pol``,
  though: ``scale_s`` always multiplies ``refl[:, 0, 0]`` (R_ss, native
  "s"); ``scale_p`` always multiplies ``refl[:, 1, 1]`` (R_pp, native
  "p"). So ``scale_s``/``scale_p`` map straight across to refloxide's
  ``scale_s``/``scale_p`` -- no relabeling -- but ``theta_offset_s``/
  ``theta_offset_p`` (selected via the SAME pol string as the R_pp/R_ss
  output swap) need the label swap when mapped onto refloxide's
  ``theta_offset_s``/``theta_offset_p``. Getting this backwards for
  either one is a real, easy-to-make trap -- confirmed by trial in
  building the refloxide comparison example: swapping the wrong one
  reproduces the legacy curve to ~5-20% relative error (looks plausible!)
  rather than machine precision.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

PICKLE_PATH = (
    Path.home() / "projects/refl-analysis/@models/xrr/znpc/dft/dft_en_offset_new2.pkl"
)
OUT_DIR = PICKLE_PATH.parent
SUMMARY_PATH = OUT_DIR / "dft_en_offset_new2_summary.json"
DATA_PATH = OUT_DIR / "dft_en_offset_new2_data.parquet"

ANCHOR_ENERGY = 283.7

print(f"loading {PICKLE_PATH} ...", flush=True)
with open(PICKLE_PATH, "rb") as f:
    global_objective = pickle.load(f)
print(
    f"loaded: {type(global_objective)}, {len(global_objective.objectives)} objectives",
    flush=True,
)

objectives = sorted(global_objective.objectives, key=lambda o: float(o.model.energy))
energies = [float(o.model.energy) for o in objectives]

# %% Anchor structure geometry (every other energy's slab params are refnx
# constraints pointing at this one -- extracting the anchor gets the true,
# independent values directly).

anchor = next(o for o in objectives if float(o.model.energy) == ANCHOR_ENERGY)
structure = anchor.model.structure

layers = []
for slab in structure:
    sld = slab.sld
    # NOTE: hasattr(sld, "rotation") is NOT a reliable uniaxial/isotropic
    # check -- pyref's base Scatterer.__init__ sets a default zero
    # `self.rotation = Parameter()` on EVERY scatterer type, including
    # isotropic MaterialSLD, which never updates or uses it. Check the
    # concrete class instead.
    is_uniaxial = type(sld).__name__ == "UniTensorSLD"
    layer = {
        "name": slab.name.rsplit("_", 1)[0],  # strip the "_283.7" energy suffix
        "kind": "uniaxial" if is_uniaxial else "isotropic",
        "thick": float(slab.thick.value or 0.0),
        "rough": float(slab.rough.value or 0.0),
        "density": float(sld.density.value or 0.0),
    }
    if is_uniaxial:
        layer["rotation"] = float(sld.rotation.value or 0.0)
    else:
        layer["formula"] = getattr(sld, "formula", "")
    layers.append(layer)
    print(f"  layer: {layer}", flush=True)

# %% Per-energy instrument corrections

corrections = []
for o in objectives:
    m = o.model
    corrections.append(
        {
            "energy": float(m.energy),
            "scale_s": float(m.scale_s.value or 0.0),
            "scale_p": float(m.scale_p.value or 0.0),
            "theta_offset_s": float(m.theta_offset_s.value or 0.0),
            "theta_offset_p": float(m.theta_offset_p.value or 0.0),
            "bkg": float(m.bkg.value or 0.0),
            "dq": float(m.dq.value or 0.0),
            "q_offset": float(m.q_offset.value or 0.0),
            "energy_offset": float(m.energy_offset.value or 0.0),
        }
    )

energy_offsets = {c["energy_offset"] for c in corrections}
print(f"distinct energy_offset values across all energies: {energy_offsets}", flush=True)

summary = {
    "source_pickle": str(PICKLE_PATH),
    "anchor_energy": ANCHOR_ENERGY,
    "ooc_csv": str(Path.home() / "projects/refl-analysis/@models/optical/znpc/dft.csv"),
    "layers": layers,
    "corrections": corrections,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
print(f"wrote {SUMMARY_PATH}", flush=True)

# %% Measured data + legacy model predictions, per energy per native channel.
#
# data.s is the FIRST concatenated chunk, data.p the SECOND -- these are
# pyref's LEGACY labels, not native ones. See the module docstring for the
# confirmed pol/scale/theta_offset label mapping.

rows = {"energy": [], "pol": [], "q": [], "r": [], "r_err": [], "legacy_pred": []}
for o in objectives:
    m = o.model
    data = o.data
    e = float(m.energy)

    # legacy .s chunk -> native pol "p" == R_pp == model.model() with pol="s"
    q_native_p = np.asarray(data.s.x)
    r_native_p = np.asarray(data.s.y)
    err_native_p = np.asarray(data.s.y_err)
    m.pol = "s"
    pred_native_p = np.asarray(m.model(q_native_p))

    # legacy .p chunk -> native pol "s" == R_ss == model.model() with pol="p"
    q_native_s = np.asarray(data.p.x)
    r_native_s = np.asarray(data.p.y)
    err_native_s = np.asarray(data.p.y_err)
    m.pol = "p"
    pred_native_s = np.asarray(m.model(q_native_s))

    for pol, q, r, err, pred in (
        ("p", q_native_p, r_native_p, err_native_p, pred_native_p),
        ("s", q_native_s, r_native_s, err_native_s, pred_native_s),
    ):
        n = len(q)
        rows["energy"].extend([e] * n)
        rows["pol"].extend([pol] * n)
        rows["q"].extend(q.tolist())
        rows["r"].extend(r.tolist())
        rows["r_err"].extend(err.tolist())
        rows["legacy_pred"].extend(pred.tolist())

    print(
        f"E={e:.1f}: native p n={len(q_native_p)}, native s n={len(q_native_s)}",
        flush=True,
    )

frame = pl.DataFrame(rows)
frame.write_parquet(DATA_PATH)
print(f"wrote {DATA_PATH} ({len(frame)} rows)", flush=True)
print("DONE", flush=True)
