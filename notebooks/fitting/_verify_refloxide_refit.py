"""Mirror refloxide_refit.ipynb for non-interactive verification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.pyref_patch import pyref_patched
from utils import read_fit, read_ooc
from utils.refloxide_fitting import compare_kernel_paths, convert_dft_fit_bundle


def main() -> None:
    if pyref_patched():
        msg = (
            "Run in a fresh process without prior patch_pyref / utils.models import"
        )
        raise RuntimeError(msg)

    bundle = read_fit("dft/dft_en_offset_new2.pkl", material="znpc", source="local")
    ooc_df = read_ooc("dft.csv", material="znpc")
    stock_logl = float(bundle.logl())
    print(f"n_objectives={len(bundle.objectives)}")
    print(f"pyref_stock_logl={stock_logl:.6f}")

    _model, batched = convert_dft_fit_bundle(bundle, ooc_df)
    print(f"n_batched_terms={len(batched.terms)}")
    print(f"n_varying={len(batched.varying_parameters())}")

    result = compare_kernel_paths(bundle, batched, parallel=False)
    summary = pd.DataFrame(
        {
            "path": ["pyref_stock", "refloxide_python", "refloxide_rust"],
            "logl": [
                result.pyref_stock_logl,
                result.refloxide_python_logl,
                result.refloxide_rust_logl,
            ],
        }
    )
    print(summary.to_string(index=False))
    print(f"python_rust_logl_delta={result.python_rust_logl_delta:.6e}")
    print(
        "max_abs_reflectivity_delta_python_rust="
        f"{result.max_abs_reflectivity_delta_python_rust:.6e}"
    )
    print("reflectivity_head:")
    head = pd.DataFrame(
        {
            "pyref_stock": result.pyref_stock_reflectivity[:8],
            "refloxide_python": result.refloxide_python_reflectivity[:8],
            "refloxide_rust": result.refloxide_rust_reflectivity[:8],
        }
    )
    print(head.to_string())
    agree = result.python_rust_logl_delta < 1e-9 * max(1.0, abs(result.refloxide_python_logl))
    print(f"python_rust_agree={agree}")


if __name__ == "__main__":
    main()
