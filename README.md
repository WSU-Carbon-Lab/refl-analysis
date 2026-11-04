# refl-analysis

Analysis code for *"Beyond the diffraction limit to track molecular states at buried interfaces"*
(resonant X-ray reflectivity of ZnPc thin films; free-tensor vs. ML-DFT slab models).

Repository: <https://github.com/WSU-Carbon-Lab/refl-analysis> · License: MIT (see `LICENSE`)

## Contents

| Path | What |
|---|---|
| `demo/` | Small **synthetic** dataset + script that runs in seconds (start here) |
| `src/utils/` | Shared code (I/O, fitting helpers, plotting) |
| `notebooks/manuscript/` | Notebooks that generate each manuscript / SI figure |
| `notebooks/fitting/`, `scripts/` | Fitting workflows |

Reflectivity physics (Parratt recursion, anisotropic tensor model, fitting) is described in the
manuscript Methods and Supplementary Information; implementation is in the [`refloxide`](https://pypi.org/project/refloxide/) package, which is the intended interface for all reflectivity modeling and fitting here.

## Requirements

- Python 3.12 · [uv](https://docs.astral.sh/uv/)
- Core dependencies (exact versions in `uv.lock`): `refloxide` 0.2.1 (prebuilt wheels for Linux, macOS, Windows), `refnx`, numpy, scipy, pandas, polars, matplotlib
- Tested on: macOS 26 (Apple M-series), Python 3.12. No non-standard hardware needed
  (a GPU is optional and only used by some fitting scripts).
- **Optional legacy dependency:** `pyref` is needed only for the manuscript/fitting notebooks and `src/utils`
  code written against `pyref.fitting` (and to unpickle the original fit files). Install with
  `uv sync --extra legacy`. It has a prebuilt wheel for Linux x86_64 only; elsewhere it compiles from source
  and needs a Rust toolchain (<https://rustup.rs>). The demo does **not** need it.
- Typical install time: ~5 min on a normal desktop (`uv sync`; longer if `pyref` must compile with `--extra legacy`).

## Run the demo

```bash
uv sync
uv run python demo/run_demo.py   # needs only refloxide + refnx
```

Expected output (~3 s) — recovers the simulated film (150 Å, 1.5 g/cm³, 4 Å roughness):

```
thickness = 149.9 A, density = 1.49 g/cm3, roughness = 3.9 A, chi2/N = 0.73
```

`demo/make_demo_data.py` regenerates `demo/demo_xrr.csv`.

## Run on your own data

```bash
uv run python demo/run_demo.py my_data.csv   # columns: q [1/Å], R, R_err
```

Edit `ENERGY_EV` and the layer stack in `demo/run_demo.py` for your sample.

## Reproducing the manuscript figures

The manuscript notebooks use legacy `pyref`-based helpers: run `uv sync --extra legacy` first.
New work should use `refloxide.model` / `refloxide.objective` directly (see `demo/run_demo.py`).

The manuscript notebooks read measured data and fit results from `@data/` and `@models/`.
**These are not distributed with the repository; they are available from the authors on request.**
With the files placed under those folders (layout: `@data/xrr/znpc/`, `@models/xrr/znpc/`,
`@models/optical/znpc/`), run the notebooks in `notebooks/manuscript/` (`FIG1`–`FIG4`, `SI*`, `APX*`).

### Raw-data examples

Raw CCD detector images from the 2024 November beamtime are preserved in this repository's git history
(e.g. `git show 067728c:"data/2024Nov/2024 11 16/CCD Scan 85272/CCD/<file>"`; list with
`git ls-tree -r --name-only 067728c -- data/2024Nov`). The reduced reflectivity, fits, and
2026 beamtime data are available from the authors on request.

## Notebook index (`notebooks/manuscript/`)

| Notebook | Produces | Needs (on request) |
|---|---|---|
| `FIG1` | Fig. 1 optical tensor, DFT vs. experiment | `@models/optical/znpc/{dft,exp}.csv` |
| `FIG2` | Fig. 2 free-tensor vs. DFT fits | XRR data, `dft/` and `free/` fit pickles, `dft.csv` |
| `FIG3` | Fig. 3 orientation | same fits + refloxide refit |
| `FIG4` | Fig. 4 and SI 12 graded profiles (MD strip) | fits, XRR data, MD profiles (`.npz`) |
| `SI2` | SI 2 GIWAXS reduction | Xeuss GIWAXS `.edf`/CSV |
| `SI9`, `SI10`, `SI11`, `SI13` | SI 9–11, 13 | fits / DFT tensors / unit cells |
| `APX1`, `APX3`–`APX6` | Appendix figures | fits / scans |

Fitting workflows that produce the fit files are in `notebooks/fitting/` and `scripts/`.
Beamtime reduction (`notebooks/data_collection/`) expects raw beamline data at a local path set in the first cell.
