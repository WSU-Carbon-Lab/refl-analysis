import pickle
from pathlib import Path

import numpy as np
import pyref.fitting as fit

GRADED = Path("@models/xrr/znpc/graded/graded_fit.pkl").resolve()
DFT = Path("@models/xrr/znpc/dft/dft_en_offset_new2.pkl").resolve()


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def describe(name, obj):
    print("=" * 70)
    print(name, "->", type(obj))
    objectives = []
    if isinstance(obj, fit.GlobalObjective):
        objectives = list(obj.objectives)
        print("  GlobalObjective with", len(objectives), "objectives")
    else:
        objectives = [obj]
    for o in objectives:
        m = o.model
        struct = m.structure
        try:
            energy = float(m.energy)
        except Exception as e:
            energy = f"?({e})"
        pol = getattr(m, "pol", None)
        print(f"  - obj energy={energy} pol={pol} structure={type(struct).__name__}")
        comps = list(struct.components)
        print(f"    components ({len(comps)}):")
        for i, c in enumerate(comps):
            print(f"      [{i}] {type(c).__name__} name={getattr(c,'name','')!r}")
    return objectives


g = load(GRADED)
d = load(DFT)
gobjs = describe("GRADED", g)
dobjs = describe("DFT", d)


def dump_materialized(o, energy):
    m = o.model
    struct = m.structure
    print(f"\n  materializing {type(struct).__name__} at energy={energy}")
    for meth in ("slabs", "structure_slabs"):
        if hasattr(struct, meth):
            try:
                s = getattr(struct, meth)()
                arr = np.asarray(s)
                print(f"    {meth}() -> shape {arr.shape}")
                print(arr)
                break
            except Exception as e:
                print(f"    {meth}() failed: {e}")


print("\n\n### energies available")
print("graded:", [float(o.model.energy) for o in gobjs])
print("dft:", [float(o.model.energy) for o in dobjs])
