import pickle
from pathlib import Path

import numpy as np

from utils.graded_objective import (
    extract_graded_fit_context,
    structure_with_upsampled_film,
)

ENERGY = 283.7
with open(Path("@models/xrr/znpc/graded/graded_fit.pkl").resolve(), "rb") as fh:
    graded = pickle.load(fh)

ctx = extract_graded_fit_context(graded)
print("film num_slabs", ctx.film.num_slabs, "total_thick", float(ctx.film.total_thick.value))
print("alpha vac/bulk/si (deg):",
      np.degrees([ctx.film.alpha_vac.value, ctx.film.alpha_bulk.value, ctx.film.alpha_si.value]))

up = structure_with_upsampled_film(ctx.structure, num_slabs=120)
sl = np.asarray(up.slabs(), dtype=np.float64)
tn = np.asarray(up.tensor(energy=ENERGY), dtype=np.complex128)
print("upsampled layers", sl.shape, "tensor", tn.shape)

from utils.graded_objective import bookended_film_from_structure

film_up = bookended_film_from_structure(up)
z = film_up.mid_points
print("orientation range deg:", np.degrees(film_up.orientation(z)).min(), np.degrees(film_up.orientation(z)).max())
