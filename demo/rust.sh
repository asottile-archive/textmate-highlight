#!/usr/bin/env bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
exec python3 "${HERE}/../highlight_demo.py" \
    highlight \
    "${HERE}/themes/dark_plus_vs.json" \
    "${HERE}/languages/rust.plist" \
    "${HERE}/files/part2.rs"
