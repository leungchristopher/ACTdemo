#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

exec uv run --locked python -m so100_flute.corrections "$@"
