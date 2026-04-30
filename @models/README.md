# Models Folder Contract

This repository treats `@models/` as a local mirror of Hugging Face model repositories.

Directory convention:

- `@models/<model-type>/<material>`

Hugging Face convention:

- `models/carbon-lab/<model-type>-<material>`

Use `scripts/hf_sync.py` or `make hf-*` targets to pull from and push to Hugging Face.

Only scaffold directories and README files should be committed to GitHub. Artifact files live on Hugging Face.
