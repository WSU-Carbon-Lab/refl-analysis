# XRR Notebooks

Repository for X-ray Reflectivity (XRR) data analysis, fitting, and related research workflows.

## Structure

This repository follows a code-first structure with Hugging Face-backed artifact storage:

- `**@data/**` - Local mirror root for Hugging Face datasets
- `**@models/**` - Local mirror root for Hugging Face model repos
- `**notebooks/**` - Jupyter notebooks organized by subproject
- `**src/**` - Reusable Python code and utilities
- `**scripts/**` - Standalone executable scripts
- `**reports/**` - Generated outputs (figures, tables, PDFs)
- `**configs/**` - Project configuration, including HF artifact mapping
- `**references/**` - External reference materials
- `**docs/**` - Documentation

## Hugging Face Artifact Mapping

The repository uses deterministic path-to-repo mapping for datasets and models.

### Data mapping

- Local: `@data/<experiment-type>/<material>`
- Hub: `datasets/carbon-lab/<experiment-type>-<material>`

Examples:

- `@data/xrr/znpc` -> `datasets/carbon-lab/xrr-znpc`
- `@data/xrr/photoresist` -> `datasets/carbon-lab/xrr-photoresist`

### Model mapping

- Local: `@models/<model-type>/<material>`
- Hub: `models/carbon-lab/<model-type>-<material>`

Examples:

- `@models/xrr/znpc` -> `models/carbon-lab/xrr-znpc`
- `@models/optical/znpc` -> `models/carbon-lab/optical-znpc`

### Configuration

Mappings are defined in:

- `configs/hf-artifacts.toml`

This config is the single source of truth for sync operations.

### Sync tooling

- Script: `scripts/hf_sync.py`
- Make targets:
  - `make hf-plan`
  - `make hf-validate`
  - `make hf-check-remote-all`
  - `make hf-pull-all`
  - `make hf-push-all`
- Targeted sync:
  - `make hf-check-remote-target TARGET=@data/xrr/znpc`
  - `make hf-pull-target TARGET=@data/xrr/znpc`
  - `make hf-push-target TARGET=@models/xrr/znpc`
- Dry run:
  - `make hf-pull-all DRY_RUN=--dry-run`
  - `make hf-check-remote-all DRY_RUN=--dry-run`

Run these before analysis and before publishing updated artifacts.

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

### Bootstrap

After cloning, run bootstrap to validate HF mappings and download configured artifacts:

```bash
make bootstrap
```

Dry run:

```bash
make bootstrap-dry-run
```

### Directory Navigation

- **Notebooks**: Organized by subproject in `notebooks/[subproject]/`
- **Data mirrors**: `@data/<experiment-type>/<material>`
- **Model mirrors**: `@models/<model-type>/<material>`
- **Figures**: All figures centralized in `reports/figures/[subproject]/`
- **Code**: Reusable modules in `src/`, scripts in `scripts/`

## Development

This repository uses:

- **ruff** for linting and code formatting
- **uv** for dependency management
- **Python 3.12+** with type hints

Code style follows PEP 8 with adaptations for scientific computing (e.g., uppercase variables for physical constants).

## Project History

This repository was reorganized in January 2026 from a flat structure to the current multi-project organization. All file moves preserved Git history using `git mv`.