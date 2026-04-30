# Data Folder Contract

This repository treats `@data/` as a local mirror of Hugging Face dataset repositories.

Directory convention:

- `@data/<experiment-type>/<material>`

Hugging Face convention:

- `datasets/carbon-lab/<experiment-type>-<material>`

Use `scripts/hf_sync.py` or `make hf-*` targets to pull from and push to Hugging Face.

Only scaffold directories and README files should be committed to GitHub. Artifact files live on Hugging Face.