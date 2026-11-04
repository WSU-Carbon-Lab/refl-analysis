"""Generate the small synthetic demo dataset (demo/demo_xrr.csv).

Simulates s-polarised reflectivity of a ZnPc film on SiO2/Si at 283.7 eV with
5% multiplicative noise. NOT real data; real data/fits are available on request.
"""
from pathlib import Path

import numpy as np
from pyref.fitting import MaterialSLD, ReflectModel

E = 283.7
rng = np.random.default_rng(42)
mats = {n: MaterialSLD(f, density=d, energy=E, name=n) for n, f, d in
        [("air", "", 0), ("znpc", "C32H16N8Zn", 1.5), ("sio2", "SiO2", 2.2), ("si", "Si", 2.33)]}
s = mats["air"](0, 0) | mats["znpc"](150, 4) | mats["sio2"](20, 3) | mats["si"](0, 2)
q = np.linspace(0.01, 0.25, 150)
r = ReflectModel(s, energy=E, pol="s")(q)
dr = 0.05 * r
out = np.column_stack([q, r + rng.normal(0, dr), dr])
np.savetxt(Path(__file__).with_name("demo_xrr.csv"), out, delimiter=",",
           header="q_invA,R,R_err", comments="")
