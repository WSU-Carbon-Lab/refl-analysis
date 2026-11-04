"""Fit the demo dataset and print recovered film parameters (~seconds on a laptop).

Usage:  python demo/run_demo.py [path/to/your_data.csv]
Input CSV columns: q (1/Angstrom), R, R_err.  Set ENERGY_EV to your photon energy.
Truth for the bundled demo: thickness 150 A, density 1.5 g/cm3, roughness 4 A.
"""
import sys
from pathlib import Path

import numpy as np
from pyref.fitting import CurveFitter, MaterialSLD, Objective, ReflectModel
from refnx.dataset import Data1D

ENERGY_EV = 283.7
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("demo_xrr.csv")
q, R, dR = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)

mk = lambda f, d, n: MaterialSLD(f, density=d, energy=ENERGY_EV, name=n)
air, film, ox, si = mk("", 0, "air"), mk("C32H16N8Zn", 1.2, "ZnPc"), mk("SiO2", 2.2, "SiO2"), mk("Si", 2.33, "Si")
s = air(0, 0) | film(120, 3) | ox(20, 3) | si(0, 2)
for prm in s.parameters.flattened():
    prm.vary = False  # fit only the ZnPc film
film.density.setp(vary=True, bounds=(0.8, 2.0))
s[1].thick.setp(vary=True, bounds=(50, 300))
s[1].rough.setp(vary=True, bounds=(0, 15))

obj = Objective(ReflectModel(s, energy=ENERGY_EV, pol="s"), Data1D((q, R, dR)))
CurveFitter(obj).fit("differential_evolution", seed=1)
print(f"thickness = {s[1].thick.value:.1f} A, density = {film.density.value:.2f} g/cm3, "
      f"roughness = {s[1].rough.value:.1f} A, chi2/N = {obj.chisqr() / len(q):.2f}")
