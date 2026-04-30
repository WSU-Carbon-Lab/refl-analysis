#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN_FLAG=""

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN_FLAG="--dry-run"
fi

cd "$ROOT_DIR"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is required. Install it first."
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Not authenticated with Hugging Face CLI. Run: hf auth login"
  exit 1
fi

.venv/bin/python scripts/hf_sync.py validate
.venv/bin/python scripts/hf_sync.py check-remote --all $DRY_RUN_FLAG
.venv/bin/python scripts/hf_sync.py pull --all $DRY_RUN_FLAG
