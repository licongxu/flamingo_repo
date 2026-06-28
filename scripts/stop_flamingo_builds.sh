#!/usr/bin/env bash
# Stop in-flight FLAMINGO catalogue/y-map build jobs and their wrapper scripts.
set -euo pipefail

PATTERNS=(
  'build_y_map.py'
  'build_halo_catalogue_M500c_1e13_zlt3.py'
  'build_halo_lightcone_catalogue.py'
  'run_build_L1_m9_y_maps'
  'run_build_all_catalogues_parallel'
  'run_build_L1_m9_catalogues'
  'run_build_L2p8_m9'
  'run_build_flamingo_sequential'
  'wait_for_flamingo_builds.sh'
  'wait_for_l2p8_catalogue_build.sh'
)

echo "$(date -Is) stopping FLAMINGO build jobs..."
for pat in "${PATTERNS[@]}"; do
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "  TERM: ${pat}"
    pkill -TERM -f "$pat" || true
  fi
done

sleep 5

for pat in "${PATTERNS[@]}"; do
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "  KILL: ${pat}"
    pkill -KILL -f "$pat" || true
  fi
done

sleep 1
remaining="$(pgrep -af 'build_y_map|build_halo_catalogue|run_build_L|wait_for_flamingo' 2>/dev/null | grep -v stop_flamingo || true)"
if [[ -n "${remaining}" ]]; then
  echo "still running:"
  echo "${remaining}"
  exit 1
fi

echo "$(date -Is) all FLAMINGO build jobs stopped."
