"""Generate the small synthetic demo dataset (demo/demo_xrr.csv) with refloxide.

Simulates s-polarised reflectivity of a ZnPc film on SiO2/Si at 283.7 eV with
5% noise. NOT real data; real data/fits are available from the authors on request.
"""
from pathlib import Path

import numpy as np
from refloxide.model import MaterialSLD, ReflectModel

E = 283.7
rng = np.random.default_rng(42)
mk = lambda f, d, n: MaterialSLD(f, density=d, energy=E, name=n)
air, film, ox, si = mk("", 0, "air"), mk("C32H16N8Zn", 1.5, "ZnPc"), mk("SiO2", 2.2, "SiO2"), mk("Si", 2.33, "Si")
structure = air(0, 0) | film(150, 4) | ox(20, 3) | si(0, 2)

q = np.linspace(0.01, 0.25, 150)
r = ReflectModel(structure)(q, E).s
dr = 0.05 * r
out = np.column_stack([q, r + rng.normal(0, dr), dr])
np.savetxt(Path(__file__).with_name("demo_xrr.csv"), out, delimiter=",",
           header="q_invA,R,R_err", comments="")
