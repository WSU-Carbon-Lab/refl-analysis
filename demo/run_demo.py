"""Fit the demo dataset with refloxide + refnx and print recovered film parameters (~seconds).

Usage:  python demo/run_demo.py [path/to/your_data.csv]
Input CSV columns: q (1/Angstrom), R (s-polarised), R_err.  Set ENERGY_EV to your photon energy.
Truth for the bundled demo: thickness 150 A, density 1.5 g/cm3, roughness 4 A.
"""
import sys
from pathlib import Path

import numpy as np
from refloxide.data import ReflectDataset
from refloxide.model import MaterialSLD, ReflectModel
from refloxide.objective import Objective
from refnx.analysis import CurveFitter

ENERGY_EV = 283.7
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("demo_xrr.csv")
q, R, dR = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
data = ReflectDataset(q, np.full_like(q, ENERGY_EV), np.full(q.shape, "s"), R, dR)

mk = lambda f, d, n: MaterialSLD(f, density=d, energy=ENERGY_EV, name=n)
air, film, ox, si = mk("", 0, "air"), mk("C32H16N8Zn", 1.2, "ZnPc"), mk("SiO2", 2.2, "SiO2"), mk("Si", 2.33, "Si")
structure = air(0, 0) | film(120, 3) | ox(20, 3) | si(0, 2)
model = ReflectModel(structure)

obj = Objective(model, data)  # also creates per-energy scale/bkg parameters
for prm in model.parameters.flattened():
    prm.vary = False  # fit only the ZnPc film
film.density.setp(vary=True, bounds=(0.8, 2.0))
structure[1].thick.setp(vary=True, bounds=(50, 300))
structure[1].rough.setp(vary=True, bounds=(0, 15))

CurveFitter(obj).fit("differential_evolution", seed=1)
print(f"thickness = {structure[1].thick.value:.1f} A, density = {film.density.value:.2f} g/cm3, "
      f"roughness = {structure[1].rough.value:.1f} A, chi2/N = {obj.chisqr() / len(q):.2f}")
