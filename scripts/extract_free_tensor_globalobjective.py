"""One-time extraction: pull a portable summary + data/predictions out of the
legacy "free tensor" pyref GlobalObjective pickle at
``@models/xrr/znpc/free/free_en_offset_init_2.pkl``, so refloxide's
comparison example (``refloxide/examples/free_tensor_model_comparison_repl.py``)
can load it without needing pyref/refnx-legacy installed.

Run once, inside this repo's own venv (has pyref + refnx):

    cd ~/projects/refl-analysis && uv run python scripts/extract_free_tensor_globalobjective.py

Unlike the DFT-fit pickle (see ``extract_dft_globalobjective.py``), this
model does NOT use one consistent scatterer class per layer across all 21
energies -- confirmed directly by inspecting the pickle: "Surface" and
"Contamination" are `pyref.fitting.structure.SLD` (a genuinely free,
independent diagonal tensor per energy, no OOC table) at every energy, but
"ZnPc" is `pyref.fitting.structure.UniTensorSLD` at 250 eV specifically and
`SLD` at every other energy -- apparently an artifact of how this
particular GlobalObjective was assembled, not an intentional per-energy
distinction. Rather than replicate that inconsistency (or guess at
its cause), this extraction reads each layer's own RESOLVED diagonal
tensor (``slab.sld.tensor``) at each energy directly, regardless of which
scatterer class produced it, and stores that. This is also exactly why
refloxide's ``FreeTensorSLD`` (a new addition -- see ``refloxide.model``)
represents every non-isotropic layer here: it holds one independent
diagonal tensor per registered energy with no formula connecting them,
which is the right shape for "whatever value this layer resolved to at
each energy," not just for genuinely free-fitted layers.

Only the ordinary (`tensor[0,0]`/`tensor[1,1]`) and extraordinary
(`tensor[2,2]`) diagonal entries are extracted/used: confirmed directly
from refloxide's Rust kernel (`compute_eigenstructure` in
``src/uniaxial.rs``) that the uniaxial solve only ever reads
`eps[(0,0)]`/`eps[(2,2)]` -- `tensor[1,1]` (the pyref `SLD` class's "yy")
is never consulted by either engine's uniaxial solver, so it is dropped
here without loss.

Writes, next to the source pickle:

- ``free_en_offset_init_2_summary.json`` -- per-layer geometry (thick,
  rough -- confirmed identical across all 21 energies for every layer)
  plus, for every layer, its resolved (delta_o, beta_o, delta_e, beta_e)
  at EVERY energy (not just an anchor -- unlike the DFT-fit pickle, most
  of these values genuinely differ energy to energy), plus the per-energy
  instrument corrections.
- ``free_en_offset_init_2_data.parquet`` -- measured data and the legacy
  model's own predicted curves, per energy per NATIVE polarization label.

Label note (identical to the DFT-fit extraction, confirmed the same
``pyref.fitting.model.ReflectModel`` class is used here too): ``pol='s'
-> refl[:, 1, 1]`` (R_pp, native "p"); ``pol='p' -> refl[:, 0, 0]`` (R_ss,
native "s"). ``scale_s``/``scale_p`` map straight across to refloxide's
``scale_s``/``scale_p`` (applied unconditionally on the raw matrix index,
not the pol-selected output); ``theta_offset_s``/``theta_offset_p`` need
the s/p label swap when mapped onto refloxide (selected via the same
inverted ``pol`` string that also swaps the output channel).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

PICKLE_PATH = (
    Path.home() / "projects/refl-analysis/@models/xrr/znpc/free/free_en_offset_init_2.pkl"
)
OUT_DIR = PICKLE_PATH.parent
SUMMARY_PATH = OUT_DIR / "free_en_offset_init_2_summary.json"
DATA_PATH = OUT_DIR / "free_en_offset_init_2_data.parquet"

ANCHOR_ENERGY = 283.7
ISOTROPIC_LAYERS = ("Vacuum", "Oxide", "Substrate")

print(f"loading {PICKLE_PATH} ...", flush=True)
with open(PICKLE_PATH, "rb") as f:
    global_objective = pickle.load(f)
print(
    f"loaded: {type(global_objective)}, {len(global_objective.objectives)} objectives",
    flush=True,
)

objectives = sorted(global_objective.objectives, key=lambda o: float(o.model.energy))
energies = [float(o.model.energy) for o in objectives]

anchor = next(o for o in objectives if float(o.model.energy) == ANCHOR_ENERGY)
layer_names = [slab.name.rsplit("_", 1)[0] for slab in anchor.model.structure]
print(f"layers: {layer_names}", flush=True)

# %% Per-layer geometry (thick/rough) + per-energy resolved (delta_o, beta_o,
# delta_e, beta_e) -- read directly off each energy's own structure, not
# assumed shared, since (unlike the DFT-fit pickle) most of these layers'
# optical tensors genuinely differ energy to energy.

layers = []
for name in layer_names:
    anchor_slab = next(s for s in anchor.model.structure if name in s.name)
    is_isotropic = name in ISOTROPIC_LAYERS
    layer = {
        "name": name,
        "kind": "isotropic" if is_isotropic else "free_tensor",
        "thick": float(anchor_slab.thick.value or 0.0),
        "rough": float(anchor_slab.rough.value or 0.0),
        "tensor_by_energy": {},
    }
    if is_isotropic:
        # Isotropic layers (MaterialSLD, both here and in the DFT-fit
        # pickle) have a literal density Parameter -- unlike the free
        # tensor layers (pyref's SLD class has no density concept at all,
        # just raw diagonal tensor components), so this is the one
        # quantity that can be transplanted directly between the two
        # models' isotropic layers. Density varies slightly energy to
        # energy in this fit (unlike thick/rough); anchor value only, same
        # convention as the DFT-fit extraction's anchor-energy geometry.
        layer["density"] = float(anchor_slab.sld.density.value or 0.0)
    for o in objectives:
        slab = next(s for s in o.model.structure if name in s.name)
        # sanity: geometry really is shared across energies for every layer
        assert abs(float(slab.thick.value or 0.0) - layer["thick"]) < 1e-9, name
        assert abs(float(slab.rough.value or 0.0) - layer["rough"]) < 1e-9, name
        t = slab.sld.tensor
        layer["tensor_by_energy"][str(float(o.model.energy))] = {
            "delta_o": float(t[0, 0].real),
            "beta_o": float(t[0, 0].imag),
            "delta_e": float(t[2, 2].real),
            "beta_e": float(t[2, 2].imag),
            "scatterer_type": type(slab.sld).__name__,
        }
    layers.append(layer)
    scatterer_types = {v["scatterer_type"] for v in layer["tensor_by_energy"].values()}
    print(f"  layer: {name} ({layer['kind']}), scatterer types used: {scatterer_types}")

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
    "layers": layers,
    "corrections": corrections,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
print(f"wrote {SUMMARY_PATH}", flush=True)

# %% Measured data + legacy model predictions, per energy per native channel.

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
