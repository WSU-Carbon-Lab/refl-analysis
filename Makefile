PYTHON := .venv/bin/python
SYNC := $(PYTHON) scripts/hf_sync.py
DRY_RUN ?=

.PHONY: bootstrap bootstrap-dry-run hf-plan hf-validate hf-check-remote-all hf-check-remote-target hf-pull-all hf-push-all hf-pull-target hf-push-target

bootstrap:
	./scripts/bootstrap.sh

bootstrap-dry-run:
	./scripts/bootstrap.sh --dry-run

hf-plan:
	$(SYNC) plan

hf-validate:
	$(SYNC) validate

hf-check-remote-all:
	$(SYNC) check-remote --all $(DRY_RUN)

hf-check-remote-target:
	$(SYNC) check-remote --target $(TARGET) $(DRY_RUN)

hf-pull-all:
	$(SYNC) pull --all $(DRY_RUN)

hf-push-all:
	$(SYNC) push --all $(DRY_RUN)

hf-pull-target:
	$(SYNC) pull --target $(TARGET) $(DRY_RUN)

hf-push-target:
	$(SYNC) push --target $(TARGET) $(DRY_RUN)
