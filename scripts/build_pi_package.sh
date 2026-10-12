#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ]]; then
    python "$root/scripts/build_pi_package.py" "$@"
else
    python3 "$root/scripts/build_pi_package.py" "$@"
fi
