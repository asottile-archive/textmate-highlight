#!/usr/bin/env bash
set -euo pipefail
if [ "${DEBUG:-}" = "1" ]; then
    args=(-mpdb)
else
    args=()
fi
cd "$(dirname "$(readlink -f "$0")")/.."
exec python "${args[@]}" ./run demo/files/invalid.xml
