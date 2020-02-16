#!/usr/bin/env bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
exec python3 "${HERE}/../highlight_demo.py" \
    highlight \
    "${HERE}/themes/dark_vs.json" \
    "${HERE}/languages" \
    "${HERE}/files/patch"
