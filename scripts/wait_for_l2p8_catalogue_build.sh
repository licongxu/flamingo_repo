#!/usr/bin/env bash
# Wait until no L2p8_m9 catalogue build is running.
set -euo pipefail

PATTERN='build_halo_lightcone_catalogue.py --variant L2p8_m9'

while pgrep -f "$PATTERN" >/dev/null 2>&1; do
  echo "$(date -Is) waiting for existing L2p8_m9 catalogue build to finish..."
  sleep 120
done

echo "$(date -Is) no L2p8_m9 catalogue build running."
