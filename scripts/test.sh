#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

exec uv run --locked --extra train --extra dev python -m pytest "$@"
