#!/usr/bin/env bash
# All relative data/config/output paths are resolved from this copy's root.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT"
