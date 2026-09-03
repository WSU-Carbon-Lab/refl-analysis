"""Benchmark diagnostic MCMC objective vs stock GlobalObjective rebuild."""

from __future__ import annotations

import time

from utils import (
    build_dft_diagnostic_pyref_mcmc_objective,
    read_fit,
    read_ooc,
)
from utils.models import CurveFitter, configure_refloxide_fitting

configure_refloxide_fitting(use_rust=True, parallel=False)

dft = read_fit("dft/dft_en_offset_new2.pkl", material="znpc")
free = read_fit("free/free_en_offset_init_2.pkl", material="znpc")
ooc = read_ooc("dft.csv", material="znpc")

mcmc_obj = build_dft_diagnostic_pyref_mcmc_objective(dft, ooc, free_bundle=free)
varying = mcmc_obj.varying_parameters()
print("diagnostic MCMC varying:", len(varying))

for _ in range(5):
    mcmc_obj.logl()

n = 50
start = time.perf_counter()
for _ in range(n):
    mcmc_obj.logl()
elapsed_ms = (time.perf_counter() - start) / n * 1000
print(f"logl: {elapsed_ms:.2f} ms")

fitter = CurveFitter(mcmc_obj, walkers_per_param=10)
fitter.initialise(random_state=0)
start = time.perf_counter()
fitter.sample(1, pool=1, skip_check=True, verbose=False)
print(
    f"1 emcee step pool=1: {time.perf_counter() - start:.2f} s "
    f"({fitter._nwalkers} walkers)"
)
