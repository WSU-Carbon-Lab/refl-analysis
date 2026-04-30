# XRR Notebooks

Repository for X-ray Reflectivity (XRR) data analysis, fitting, and related research workflows.

## Structure

This repository follows a Cookiecutter Data Science-inspired structure, organized for multiple research subprojects:

- **`data/`** - Data files organized by processing stage (raw/interim/processed)
- **`notebooks/`** - Jupyter notebooks organized by subproject
- **`src/`** - Reusable Python code and utilities
- **`scripts/`** - Standalone executable scripts
- **`reports/`** - Generated outputs (figures, tables, PDFs)
- **`models/`** - Saved model artifacts and checkpoints
- **`config/`** - Configuration files
- **`references/`** - External reference materials
- **`docs/`** - Documentation

## Subprojects

- **data_collection**: Beamtime data collection and processing
- **fitting**: X-ray reflectivity (XRR) fitting and analysis
- **manuscript**: Manuscript preparation and figure generation
- **optical_models**: Optical constants modeling and NEXAFS analysis
- **dft**: Density functional theory (DFT) analysis
- **giwaxs**: Grazing-incidence wide-angle X-ray scattering analysis
- **photoresist**: Photoresist-related analysis
- **training_data**: Training data generation for machine learning

## Getting Started

### Prerequisites

- Python 3.12+
- uv (package manager)

### Installation

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Directory Navigation

- **Notebooks**: Organized by subproject in `notebooks/[subproject]/`
- **Data**: Raw data in `data/raw/`, processed in `data/processed/`
- **Figures**: All figures centralized in `reports/figures/[subproject]/`
- **Code**: Reusable modules in `src/`, scripts in `scripts/`

## Documentation

- **Architecture**: See `docs/architecture.md` for detailed structure documentation
- **Notebook Index**: See `docs/notebooks_index.md` for catalog of all notebooks
- **Migration Map**: See `docs/migration_map.md` for file reorganization details

## Development

This repository uses:
- **ruff** for linting and code formatting
- **uv** for dependency management
- **Python 3.12+** with type hints

Code style follows PEP 8 with adaptations for scientific computing (e.g., uppercase variables for physical constants).

## Project History

This repository was reorganized in January 2026 from a flat structure to the current multi-project organization. All file moves preserved Git history using `git mv`.
