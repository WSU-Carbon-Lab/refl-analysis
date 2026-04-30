import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyref.fitting as fit
from refnx.analysis import CurveFitter, GlobalObjective, Objective, Transform

df = pl.read_parquet("june_processed.parquet").sort(pl.col("pol"), descending=True)
df = df.filter(pl.col("sample").str.starts_with("mono"))

data = {}
for en, g in df.group_by("Beamline Energy [eV]", maintain_order=True):
    d = fit.to_reflect_dataset(g)
    # break
    data[str(en[0])] = d
data.keys()


ooc = pd.read_csv(
    "optical_constants.csv",
)
ooc.plot(x="energy")
plt.xlim(250, 320)
plt.ylim(-6e-3, 1.25e-2)

ZNPC = "C32H16N8Zn"

# Define materials list with dictionaries
materials = [
    {"name": "Substrate", "formula": "Si", "density": None, "dims": (0, 1.5)},
    {"name": "Oxide", "formula": "SiO2", "density": 2.14658, "dims": (8, 6.38)},
    {
        "name": "Sub-Layer",
        "formula": ZNPC,
        "density": 1.93955,
        "dims": (6.08, 4.4),
        "rotation": 0,
    },
    {
        "name": "Bulk-layer",
        "formula": ZNPC,
        "density": 1.37429,
        "dims": (196.441, 7.216),
        "rotation": 2 * np.pi,
    },
    {
        "name": "Sup-layer",
        "formula": ZNPC,
        "density": 1.37429,
        "dims": (10, 5),
        "rotation": 0,
    },
    {
        "name": "Vac",
        "formula": "",
        "density": 0,
        "dims": (0, 0),
    },  # Changed empty string to 0 for density
]

vacuum = fit.MaterialSLD("", 0, name="Vac")(10, 0)
vacuum.thick.setp(vary=False)
vacuum.rough.setp(vary=False)
vacuum.sld.density.setp(vary=False)

"""
Construct Fitting for 250 eV
"""


def construct_slab(en):
    substrate = fit.MaterialSLD("Si", energy=en, name="Substrate")(10, 1.5)
    substrate.thick.setp(vary=False)
    substrate.rough.setp(vary=False)
    substrate.sld.density.setp(vary=True, bounds=(2, 3))

    oxide = fit.MaterialSLD("SiO2", 2.14658, energy=250, name="Oxide")(8, 6.38)
    oxide.thick.setp(vary=False)
    oxide.rough.setp(vary=False)
    oxide.sld.density.setp(vary=True)

    contamination = fit.UniTensorSLD(ooc, density=1.9395, energy=en, name="Sub-Layer")(
        6.08, 4.4
    )
    contamination.thick.setp(vary=True, bounds=(0, 12))
    contamination.rough.setp(vary=True, bounds=(0, 12))
    contamination.sld.density.setp(vary=True, bounds=(1, 2))
    contamination.sld.rotation.setp(vary=True, bounds=(0, np.pi / 2))

    bulk = fit.UniTensorSLD(
        ooc, density=1.37429, energy=en, rotation=np.pi / 2, name="Bulk-layer"
    )(196.441, 7.216)
    bulk.thick.setp(vary=True, bounds=(180, 200))
    bulk.rough.setp(vary=True, bounds=(0, 15))
    bulk.sld.density.setp(vary=True, bounds=(1, 1.6))
    bulk.sld.rotation.setp(vary=True, bounds=(0, np.pi / 2))

    surface = fit.UniTensorSLD(ooc, density=1.37429, energy=en, name="Sup-Layer")(
        6.08, 4.4
    )
    surface.thick.setp(vary=True, bounds=(0, 20))
    surface.rough.setp(vary=True, bounds=(0, 20))
    surface.sld.density.setp(vary=True, bounds=(1.2, 2))
    surface.sld.rotation.setp(vary=True, bounds=(0, np.pi / 2))

    stack = vacuum | surface | bulk | contamination | oxide | substrate
    stack.name = f"ZnPc - Monolayer - {en}eV"
    return stack


energies = df["Beamline Energy [eV]"].unique().to_numpy()

stacks = {str(en): construct_slab(en) for en in energies}


def construct_model(en):
    model = fit.ReflectModel(
        stacks[en], pol="sp", energy=float(en), name=f"ZnPc Mono Layer {en}eV"
    )
    # model.scale_s.setp(vary = True, bounds = (0.9, 1.2))
    # model.scale_p.setp(vary = True, bounds = (0.8, 1.1))
    # model.theta_offset_s.setp(vary = True, bounds = (-.1, .1))
    # model.theta_offset_p.setp(vary = True, bounds = (-.1, .1))
    d = data[en]
    model.bkg.value = d.data[1].min()

    obj = Objective(model, d, transform=Transform("logY"))
    # lpe = fit.LogpExtra(obj)
    # obj.logp_extra = lpe

    return obj


objs = {en: construct_model(en) for en in ["250.0", "283.7"]}
# construct the global objective
obj = GlobalObjective([o for _, o in objs.items()])
obj.plot()
print(len(obj.varying_parameters()))

# for now we will be unconstrained
for en, stack in stacks.items():
    if en == "250.0":
        continue
    stack[1].thick.setp(vary=None, constraint=stacks["250.0"][1].thick)
    stack[1].rough.setp(vary=None, constraint=stacks["250.0"][1].rough)

    stack[2].thick.setp(vary=None, constraint=stacks["250.0"][2].thick)
    stack[2].rough.setp(vary=None, constraint=stacks["250.0"][2].rough)

    stack[3].thick.setp(vary=None, constraint=stacks["250.0"][3].thick)
    stack[3].rough.setp(vary=None, constraint=stacks["250.0"][3].rough)

    stack[4].sld.density.setp(vary=None, constraint=stacks["250.0"][4].sld.density)
    stack[5].sld.density.setp(vary=None, constraint=stacks["250.0"][5].sld.density)


# from refnx._lib.emcee.moves.gaussian import GaussianMove

fitter = CurveFitter(obj, nwalkers=200, ntemps=1)
fitter.initialise("jitter")
chain = fitter.sample(1000)

obj.plot()
plt.show()
plt.plot(-fitter.logpost, c="black", lw=0.2, alpha=0.5)
plt.show()
