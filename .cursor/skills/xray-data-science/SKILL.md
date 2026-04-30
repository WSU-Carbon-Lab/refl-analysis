---
name: xray-data-science-best-practices
description: Python and Jupyter notebook performance optimization guidelines for X-ray data science (XRR, SAXS, WAXS, XRD, XAS, NEXAFS). This skill should be used when writing, reviewing, or refactoring Python data science code to ensure optimal performance patterns and correct physics-based analysis. Triggers on tasks involving X-ray data analysis, reflectivity fitting, spectroscopic analysis, or scientific computing workflows.
license: MIT
metadata:
  author: X-ray Data Science Team - Washington State University
  version: "1.0.0"
---

# X-ray Data Science Best Practices

Comprehensive performance optimization and best practices guide for Python data science projects focused on X-ray analysis (XRR, SAXS, WAXS, XRD, XAS, NEXAFS), maintained by the X-ray Data Science Team at Washington State University. Contains 60+ rules across 11 categories, prioritized by impact from critical (eliminating I/O bottlenecks, vectorization) to incremental (advanced patterns). Includes tool selection guidelines, specialized X-ray library usage, spectroscopic analysis principles, and unit/uncertainty management.

## When to Apply

Reference these guidelines when:
- Writing new Python code for X-ray data analysis
- Processing X-ray reflectivity (XRR), scattering (SAXS/WAXS), or diffraction (XRD) data
- Performing spectroscopic analysis (XAS, NEXAFS)
- Loading or transforming large scientific datasets
- Implementing numerical computations with NumPy/SciPy
- Using specialized X-ray libraries (periodictable, PyFAI, PyPXR, pyref)
- Working with Jupyter notebooks for data analysis
- Optimizing data pipelines or computational workflows
- Fitting reflectivity or spectroscopic data

## Rule Categories by Priority

| Priority | Category | Impact | Prefix |
|----------|----------|--------|--------|
| 1 | Data Loading & I/O Optimization | CRITICAL | `io-` |
| 2 | Vectorization & NumPy Optimization | CRITICAL | `numpy-` |
| 3 | Reproducibility & Code Quality | HIGH | `repro-` |
| 4 | X-ray Specialized Libraries | HIGH | `xray-` |
| 5 | Spectroscopic Analysis | HIGH | `spectro-` |
| 6 | DataFrame Libraries (Pandas/Polars/DuckDB) | HIGH | `df-` |
| 7 | Computational Efficiency | HIGH | `compute-` |
| 8 | Memory Management | MEDIUM-HIGH | `memory-` |
| 9 | Visualization Performance | MEDIUM | `viz-` |
| 10 | Notebook Organization | MEDIUM | `notebook-` |
| 11 | Units and Uncertainty Propagation | MEDIUM-HIGH | `units-` |

## Quick Reference

### 1. Data Loading & I/O Optimization (CRITICAL)

- `io-lazy-loading` - Use lazy loading for large datasets, load data on demand
- `io-batch-files` - Batch file operations using concurrent.futures
- `io-binary-formats` - Use binary formats (Parquet, HDF5, NPY) instead of text (CSV)
- `io-chunk-csv` - Process large CSV/text files in chunks
- `io-cache-processing` - Cache expensive data transformations with pickle/joblib
- `io-memory-mapping` - Use memory mapping for arrays larger than RAM

### 2. Vectorization & NumPy Optimization (CRITICAL)

- `numpy-vectorize-loops` - Eliminate Python loops with NumPy vectorization (100-1000× speedup)
- `numpy-broadcasting` - Use NumPy broadcasting for operations between arrays of different shapes
- `numpy-avoid-conversions` - Avoid repeated array conversions, work with NumPy arrays throughout
- `numpy-inplace` - Use in-place operations to reduce memory allocation
- `numpy-preallocate` - Preallocate arrays instead of growing dynamically
- `numpy-data-types` - Use appropriate data types (float32 vs float64) to reduce memory
- `numpy-einsum` - Use einsum for tensor contractions, standard vectorization for element-wise ops

### 3. Reproducibility & Code Quality (HIGH)

- `repro-random-seeds` - Set random seeds for all random number generators (np.random.seed, random.seed)
- `repro-version-pin` - Pin exact versions of all dependencies in requirements.txt
- `repro-data-provenance` - Document data provenance (source, date, processing steps)
- `repro-type-hints` - Use type hints (NDArray, float) for function signatures
- `repro-validate-inputs` - Validate input data with assertions or informative errors
- `repro-logging` - Use logging module instead of print statements

### 4. X-ray Specialized Libraries (HIGH)

- `xray-periodictable` - Use periodictable for elemental properties, verify units explicitly
- `xray-pyfai-geometry` - Use PyFAI for diffraction integration, ensure consistent units (meters)
- `xray-reflectivity-physics` - Understand Parratt recursion and Fresnel coefficients before using PyPXR/pyref
- `xray-resolution-smearing` - Always account for instrument resolution smearing in reflectivity
- `xray-layer-constraints` - Validate physical constraints on layer parameters (thickness > 0, roughness < thickness/2)

### 5. Spectroscopic Analysis (HIGH)

- `spectro-nexafs-polarization` - Account for polarization and incident angle in NEXAFS analysis
- `spectro-orientation-fitting` - Use polarization-dependent absorption: mu(alpha) = mu_0 * [1 + beta * P2(cos(alpha))]
- `spectro-dichroism-ratio` - Calculate dichroism ratio: (mu_parallel - mu_perp) / (mu_parallel + 2*mu_perp)
- `spectro-normalization` - Normalize by background subtraction (pre-edge) and post-edge normalization
- `spectro-polarization-factor` - Account for partial beamline polarization (typically 95%)

### 6. DataFrame Libraries (HIGH)

- `df-pandas-small` - Use Pandas for data < 10 GB with rich ecosystem needs
- `df-polars-medium` - Use Polars with lazy evaluation for 10+ GB datasets
- `df-duckdb-large` - Use DuckDB for 100+ GB datasets or direct Parquet/CSV querying
- `df-categorical` - Convert string columns with repeated values to categorical dtype (50-80% memory reduction)
- `df-no-iterrows` - Never use iterrows/itertuples, use vectorized operations instead
- `df-groupby-agg` - Use single groupby with agg() instead of multiple group operations
- `df-concat-once` - Collect DataFrames in list, concat once instead of repeated concat

### 7. Computational Efficiency (HIGH)

- `compute-numba-loops` - Use Numba JIT for CPU-bound numerical loops (10-100× speedup)
- `compute-pytorch-gpu` - Use PyTorch for GPU acceleration or automatic differentiation
- `compute-multiprocessing` - Use multiprocessing for embarrassingly parallel tasks (N× speedup on N cores)
- `compute-scipy` - Use SciPy optimized functions (curve_fit, find_peaks, interp1d) instead of manual implementation
- `compute-lru-cache` - Cache expensive pure functions with functools.lru_cache
- `compute-generators` - Use generator expressions for large sequences to reduce memory

### 8. Memory Management (MEDIUM-HIGH)

- `memory-delete-variables` - Delete large unused variables and force garbage collection
- `memory-context-managers` - Always use 'with' statements for file operations
- `memory-monitor-usage` - Monitor memory usage with psutil to prevent OOM errors
- `memory-generator-pipelines` - Use generators for large data pipelines (constant memory usage)
- `memory-dataframe-copies` - Use inplace=True or avoid unnecessary DataFrame copies

### 9. Visualization Performance (MEDIUM)

- `viz-downsample-plotting` - Downsample data to screen resolution (~2000 points) for plotting
- `viz-rasterize-dense` - Rasterize plots with many data points to avoid huge vector files
- `viz-reuse-figures` - Reuse figure objects instead of creating new ones in loops
- `viz-appropriate-types` - Use hexbin or 2D histogram for dense data instead of scatter
- `viz-batch-updates` - Batch plot updates before redrawing

### 10. Notebook Organization (MEDIUM)

- `notebook-separate-loading` - Separate data loading from analysis in different cells
- `notebook-cell-magic` - Use cell magic (%%time, %lprun, %memit) for profiling
- `notebook-clear-outputs` - Clear outputs before committing notebooks to version control
- `notebook-parameters` - Use papermill to parameterize notebooks for batch processing
- `notebook-extract-modules` - Extract reusable code to Python modules, use autoreload

### 11. Units and Uncertainty Propagation (MEDIUM-HIGH)

- `units-pint` - Use Pint for explicit unit management to prevent conversion errors
- `units-uncertainties` - Use uncertainties package for automatic error propagation
- `units-combine` - Combine Pint and uncertainties for complete error handling with units
- `units-correlated` - Account for correlated uncertainties using covariance matrices
- `units-report-formatting` - Format results with uncertainties properly (ufloat formatting)

## X-Ray Data Booklet Reference

The [X-Ray Data Booklet](https://xdb.lbl.gov/) from Lawrence Berkeley National Laboratory provides essential reference data:

- **X-Ray Properties of Elements**: Electron binding energies, emission energies, fluorescence yields, Auger energies
- **Atomic Scattering Factors**: Fundamental data for reflectivity and scattering calculations
- **Mass Absorption Coefficients**: Critical for attenuation corrections
- **Synchrotron Radiation**: Characteristics and properties of synchrotron sources
- **Scattering Processes**: X-ray scattering from electrons and atoms
- **Optics and Detectors**: Crystal elements, mirrors, gratings, monochromators, detectors

Always reference the X-Ray Data Booklet when:
- Looking up atomic scattering factors or form factors
- Calculating absorption coefficients
- Understanding synchrotron radiation properties
- Determining electron binding energies for spectroscopic analysis
- Selecting appropriate optics or detector specifications

## Key Physics Principles

Understanding these principles is essential before using analysis libraries:

### Reflectivity (XRR)
- **Parratt recursive formalism**: R(j+1) = (r(j,j+1) + R(j) * exp(2*i*q(j)*d(j))) / (1 + r(j,j+1)*R(j)*exp(2*i*q(j)*d(j)))
- **Fresnel coefficients**: r = (q1 - q2)/(q1 + q2) where q = sqrt(k^2 - 4*pi*SLD)
- **Roughness effects**: Nevot-Croce factor accounts for interface roughness
- **Instrument resolution**: Angular divergence and wavelength spread must be convolved

### Spectroscopic Analysis (NEXAFS/XAS)
- **Electric dipole approximation**: sigma ~ |<i|epsilon·r|f>|^2
- **Polarization dependence**: mu(alpha) = mu_0 * [1 + beta * P2(cos(alpha))]
- **Transition moments**: pi* transitions (sin^2(alpha)), sigma* transitions (cos^2(alpha))
- **Orientation determination**: Dichroism ratio and Legendre polynomial fitting

### Diffraction (XRD)
- **Solid angle correction**: dOmega = sin(2theta) d(2theta) d(phi)
- **Polarization factors**: Critical for synchrotron radiation
- **Geometric transformations**: Detector coordinates to reciprocal space

## How to Use

Read the full [AGENTS.md](../AGENTS.md) document for detailed explanations and code examples:

Each rule includes:
- Impact assessment (CRITICAL, HIGH, MEDIUM, LOW)
- Incorrect code example with explanation
- Correct code example with explanation
- Additional context and best practices
- Performance metrics where applicable

## Common Pitfalls

### Library Usage Without Understanding
- **Incorrect**: Using PyPXR/pyref without understanding Parratt recursion
- **Correct**: Understand the physics, then use libraries as tools

### Unit Inconsistencies
- **Incorrect**: Mixing Angstroms and meters in PyFAI geometry
- **Correct**: Use consistent unit system throughout (prefer meters for PyFAI)

### Missing Instrument Effects
- **Incorrect**: Fitting reflectivity without resolution smearing
- **Correct**: Always include angular divergence and wavelength spread

### Memory Issues with Large Data
- **Incorrect**: Loading 100 GB file entirely into memory
- **Correct**: Use DuckDB to query, Polars with lazy evaluation, or chunked processing

### Incorrect Spectroscopic Normalization
- **Incorrect**: Using raw absorption without proper normalization
- **Correct**: Background subtract pre-edge, normalize to post-edge

## Full Compiled Document

For the complete guide with all rules expanded: `AGENTS.md`

## References

- [X-Ray Data Booklet](https://xdb.lbl.gov/) - Lawrence Berkeley National Laboratory
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Periodic Table Python Package](https://periodictable.readthedocs.io/en/latest/)
- [PyFAI Documentation](https://pyfai.readthedocs.io/en/stable/)
- [PyPXR Documentation](https://p-rsoxr.readthedocs.io/en/latest/)
- [NEXAFS Spectroscopy Reference](https://www.sciencedirect.com/science/article/pii/S0014305716300179)
