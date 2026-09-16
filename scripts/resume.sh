#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 CHECKPOINT_TRAIN_CONFIG [training options]" >&2
    echo "Example: $0 outputs/act_flute/checkpoints/last/pretrained_model/train_config.json --device mps" >&2
    exit 2
fi
checkpoint_config=$1
shift
if [[ ! -f "$checkpoint_config" ]]; then
    echo "Checkpoint configuration does not exist: $checkpoint_config" >&2
    exit 2
fi
exec "$PROJECT_ROOT/scripts/train.sh" --resume "$checkpoint_config" "$@"
