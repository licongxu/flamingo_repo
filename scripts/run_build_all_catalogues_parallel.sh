#!/usr/bin/env bash
# Build all FLAMINGO halo-lightcone catalogues with bounded parallelism.
# hdfstream serialises server-side: >4 concurrent jobs mostly queue, not speed up.
# L1_m9: 9 variants; L2p8_m9: lightcones 0..7 (skips complete / already-running).
set -euo pipefail

MAX_PARALLEL="${MAX_PARALLEL:-4}"

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate

REPO=/scratch/scratch-lxu/flamingo_repo
L1_OUT=/rds/rds-lxu/flamingo/L1_m9/catalogues
L2_BASE=/rds/rds-lxu/flamingo/L2p8_m9
LOGDIR="${REPO}/cobaya/logs"
BUILD_SCRIPT="${REPO}/scripts/build_halo_lightcone_catalogue.py"
mkdir -p "$LOGDIR" "$L1_OUT"

L1_VARIANTS=(
  L1_m9
  fgas+2sigma
  fgas-2sigma
  fgas-4sigma
  fgas-8sigma
  Mstar-1sigma
  Mstar-1sigma_fgas-4sigma
  Jet
  Jet_fgas-4sigma
)

catalogue_running() {
  pgrep -f "build_halo_lightcone_catalogue.py.*--out ${1}" >/dev/null 2>&1
}

l1_status() {
  python "${BUILD_SCRIPT}" --parent L1_m9 --variant "$2" --out "$1" --status
}

l2_status() {
  python "${BUILD_SCRIPT}" --variant L2p8_m9 --observer "$2" --out "$1" --status
}

LAST_PID=""

launch_l1() {
  local variant="$1"
  local out="${L1_OUT}/halo_catalogue_M500c_1e13_zlt3_${variant}_yang26rot.csv"
  local log="${LOGDIR}/build_catalogue_L1_m9_${variant}.log"
  local yflag=(--no-y-columns)
  if [[ "${variant}" == "L1_m9" && -f "${out}" ]]; then
    yflag=()
  fi

  LAST_PID=""
  if catalogue_running "${out}"; then
    echo "skip L1 ${variant}: already running"
    return 0
  fi
  local status
  status="$(l1_status "${out}" "${variant}")"
  if [[ "${status}" == "complete" ]]; then
    echo "skip L1 ${variant}: complete"
    return 0
  fi

  echo "launch L1 ${variant} -> ${out} (status=${status})"
  python "${BUILD_SCRIPT}" \
    --parent L1_m9 \
    --variant "${variant}" \
    --out "${out}" \
    "${yflag[@]}" \
    --resume \
    >>"${log}" 2>&1 &
  LAST_PID=$!
}

launch_l2() {
  local obs="$1"
  local outdir="${L2_BASE}/lightcone${obs}/catalogues"
  local out="${outdir}/halo_catalogue_M500c_1e13_zlt3_L2p8_m9_yang26rot.csv"
  local log="${LOGDIR}/build_catalogue_L2p8_m9_lc${obs}.log"
  mkdir -p "${outdir}"

  LAST_PID=""
  if catalogue_running "${out}"; then
    echo "skip L2 lc${obs}: already running"
    return 0
  fi
  local status
  status="$(l2_status "${out}" "${obs}")"
  if [[ "${status}" == "complete" ]]; then
    echo "skip L2 lc${obs}: complete"
    return 0
  fi

  echo "launch L2 lc${obs} -> ${out} (status=${status})"
  python "${BUILD_SCRIPT}" \
    --variant L2p8_m9 \
    --observer "${obs}" \
    --out "${out}" \
    --no-y-columns \
    --resume \
    >>"${log}" 2>&1 &
  LAST_PID=$!
}

wait_for_slot() {
  while true; do
    local running=0
    for pid in "${RUNNING[@]:-}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        running=$((running + 1))
      fi
    done
    if [[ "${running}" -lt "${MAX_PARALLEL}" ]]; then
      return 0
    fi
    wait -n 2>/dev/null || true
    local next=()
    for pid in "${RUNNING[@]:-}"; do
      kill -0 "${pid}" 2>/dev/null && next+=("${pid}")
    done
    RUNNING=("${next[@]}")
  done
}

track_pid() {
  RUNNING+=("$1")
}

reap_finished() {
  local next=()
  for pid in "${RUNNING[@]:-}"; do
    kill -0 "${pid}" 2>/dev/null && next+=("${pid}")
  done
  RUNNING=("${next[@]}")
}

RUNNING=()
fail=0
LAUNCHED=0

echo "$(date -Is) === catalogue build start (slab-read, max_parallel=${MAX_PARALLEL}) ==="

for variant in "${L1_VARIANTS[@]}"; do
  wait_for_slot
  launch_l1 "${variant}"
  if [[ -n "${LAST_PID}" ]]; then
    track_pid "${LAST_PID}"
    LAUNCHED=$((LAUNCHED + 1))
  fi
done

for obs in $(seq 0 7); do
  wait_for_slot
  launch_l2 "${obs}"
  if [[ -n "${LAST_PID}" ]]; then
    track_pid "${LAST_PID}"
    LAUNCHED=$((LAUNCHED + 1))
  fi
done

if [[ "${LAUNCHED}" -eq 0 ]]; then
  echo "no catalogue jobs launched (all complete or running)"
  exit 0
fi

echo "launched ${LAUNCHED} job(s), waiting (max ${MAX_PARALLEL} concurrent)..."

while [[ ${#RUNNING[@]} -gt 0 ]]; do
  if ! wait -n; then
    echo "catalogue job failed"
    fail=1
  fi
  reap_finished
done

if [[ "${fail}" -ne 0 ]]; then
  echo "$(date -Is) === catalogue build finished with errors ==="
  exit 1
fi

echo "$(date -Is) === catalogue build done ==="
