import pickle
from pathlib import Path

import numpy as np
import pyref.fitting as fit
from refloxide.integrations.pyref import uniaxial_reflectivity

from utils.field_profile import uniaxial_field_profile

ENERGY = 283.7
GRADED = Path("@models/xrr/znpc/graded/graded_fit.pkl").resolve()
DFT = Path("@models/xrr/znpc/dft/dft_en_offset_new2.pkl").resolve()


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def dft_objective_at(glob, energy):
    for o in glob.objectives:
        if np.isclose(float(o.model.energy), energy):
            return o
    raise SystemExit(f"no dft objective at {energy}")


graded = load(GRADED)
dft_glob = load(DFT)
dft = dft_objective_at(dft_glob, ENERGY)

dft_struct = dft.model.structure
layers = np.asarray(dft_struct.slabs(), dtype=np.float64)
tensor = np.asarray(dft_struct.tensor(energy=ENERGY), dtype=np.complex128)
print("DFT layers shape", layers.shape, "tensor shape", tensor.shape)
print("layers:\n", np.round(layers, 4))

q = np.linspace(0.01, 0.27, 600)
refl, _tran = uniaxial_reflectivity(q, layers, tensor, ENERGY, use_rust=True)
r_ss = refl[:, 0, 0]
r_pp = refl[:, 1, 1]

win = (q >= 0.19) & (q <= 0.22)
q_brew = float(q[win][np.argmin(r_pp[win])])
print(f"\nBrewster (R_pp min in [0.19,0.22]) at q = {q_brew:.4f} A^-1")
print(f"  R_pp(min) = {r_pp[win].min():.3e}")

print("\nValidation: reconstruction reflectance vs kernel R")
for qt in (0.05, 0.15, q_brew, 0.25):
    refl_t, _ = uniaxial_reflectivity(
        np.array([qt]), layers, tensor, ENERGY, use_rust=True
    )
    fs = uniaxial_field_profile(qt, layers, tensor, ENERGY, "s")
    fp = uniaxial_field_profile(qt, layers, tensor, ENERGY, "p")
    print(
        f"  q={qt:.4f}  s: recon={fs.reflectance:.6e} kernel[1,1]={refl_t[0,1,1]:.6e}"
        f"  p: recon={fp.reflectance:.6e} kernel[0,0]={refl_t[0,0,0]:.6e}"
    )

print("\nGraded structure materialization check")
gs = graded.model.structure
gl = np.asarray(gs.slabs(), dtype=np.float64)
gt = np.asarray(gs.tensor(energy=ENERGY), dtype=np.complex128)
print("graded layers shape", gl.shape, "tensor shape", gt.shape)
refl_g, _ = uniaxial_reflectivity(np.array([q_brew]), gl, gt, ENERGY, use_rust=True)
fgp = uniaxial_field_profile(q_brew, gl, gt, ENERGY, "p")
fgs = uniaxial_field_profile(q_brew, gl, gt, ENERGY, "s")
print(
    f"  graded q={q_brew:.4f} s recon={fgs.reflectance:.6e} kernel[1,1]={refl_g[0,1,1]:.6e}"
    f"  p recon={fgp.reflectance:.6e} kernel[0,0]={refl_g[0,0,0]:.6e}"
)
