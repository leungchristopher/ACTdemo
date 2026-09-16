#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

exec uv run --locked --extra train python -m so100_flute.training convert "$@"
