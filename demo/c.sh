#!/usr/bin/env bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
if [ "${DEBUG:-}" = "1" ]; then
    args=(-mpdb)
else
    args=()
fi
exec python3 "${args[@]}" "${HERE}/../highlight_demo.py" \
    "${HERE}/themes/dark_plus_vs.json" \
    "${HERE}/languages" \
    "${HERE}/files/test.c"
