#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# macOS needs mjpython when launching the interactive MuJoCo viewer.
python_command=python
if [[ "$(uname -s)" == Darwin ]]; then
    for argument in "$@"; do
        if [[ "$argument" == --viewer ]]; then
            python_command=mjpython
        fi
    done
fi
exec uv run --locked "$python_command" -m so100_flute replay "$@"
