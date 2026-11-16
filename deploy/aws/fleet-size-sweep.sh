#!/usr/bin/env bash
# AWS port of deploy/gcp/fleet-size-sweep.sh: sweep the number of active (graph, loadgen) pairs,
# holding per-instance settings constant, to find where the Aerospike cluster saturates.
# Stops the whole fleet before each stage so only the target count is generating load.
#
# Usage:
#   ./fleet-size-sweep.sh [max_fleet] [duration_sec]
#
# Example (max-throughput KV mode; size KV_TTL to the cluster — see README):
#   KV_MODEL=flat BALANCES=false KV_TTL=600 MODE=kv ./fleet-size-sweep.sh 12 300
source "$(dirname "$0")/lib.sh"

MAX_FLEET="${1:-12}"
DURATION="${2:-300}"
MODE="${MODE:-paired}"
FRAUD_RATIO="${FRAUD_RATIO:-0.0000001}"
SKIP_BUILD="${SKIP_BUILD:-0}"
# Fleet sizes to test (must be <= MAX_FLEET). Override: SIZES="1 2 4 8 12"
read -r -a SIZES <<< "${SIZES:-1 2 4 6 8 10 12}"

LOG="/tmp/fleet_size_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "Fleet size sweep: sizes=${SIZES[*]} mode=${MODE} duration=${DURATION}s fraud_ratio=${FRAUD_RATIO} cluster=${AEROSPIKE_CLUSTER_NAME}" | tee "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

verify_fleet_running() {
  local count="$1" ids running
  sleep 12
  read_lines ids < <(for i in $(seq 1 "${count}"); do instance_by_name loadgen "${DEPLOY_PREFIX}-loadgen-${i}"; done)
  running="$(ssm_run 60 'pgrep -x java >/dev/null && echo JAVA_UP || echo JAVA_DOWN' "${ids[@]}" 2>/dev/null | grep -c JAVA_UP || true)"
  if [[ "${running}" != "${count}" ]]; then
    echo "ERROR: loadgen running on ${running}/${count} instance(s)" | tee -a "$LOG"
    return 1
  fi
  echo "verified ${count}/${count} loadgen instance(s) running java" | tee -a "$LOG"
}

first=1
for count in "${SIZES[@]}"; do
  if (( count > MAX_FLEET )); then echo "skip count=${count} (> max_fleet=${MAX_FLEET})" | tee -a "$LOG"; continue; fi
  echo "" | tee -a "$LOG"
  echo "========== FLEET=${count}  DURATION=${DURATION}s  $(date -u) ==========" | tee -a "$LOG"

  "${AWS_DIR}/load_runner.sh" stop "${MAX_FLEET}" loadgen 2>&1 | tee -a "$LOG"

  # Build the jar once, on the first stage.
  build=$(( first == 1 ? SKIP_BUILD : 1 )); first=0
  FRAUD_RATIO="${FRAUD_RATIO}" SKIP_BUILD="${build}" \
    "${AWS_DIR}/load_runner.sh" start "${count}" "${MODE}" 0 "${DURATION}" 64 0 2>&1 | tee -a "$LOG"

  verify_fleet_running "${count}" || {
    echo "retrying start for fleet=${count}..." | tee -a "$LOG"
    FRAUD_RATIO="${FRAUD_RATIO}" SKIP_BUILD=1 \
      "${AWS_DIR}/load_runner.sh" start "${count}" "${MODE}" 0 "${DURATION}" 64 0 2>&1 | tee -a "$LOG"
    verify_fleet_running "${count}" || { echo "FATAL: fleet=${count} failed verification after retry — aborting" | tee -a "$LOG"; exit 1; }
  }
  wait_sec=$((DURATION + 15))
  echo "waiting ${wait_sec}s for run to finish..." | tee -a "$LOG"
  sleep "${wait_sec}"

  echo "--- report (fleet=${count}) ---" | tee -a "$LOG"
  "${AWS_DIR}/load_runner.sh" report "${count}" 2>&1 | tee -a "$LOG" || echo "WARN: report failed for fleet=${count}" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "Fleet size sweep complete at $(date -u)" | tee -a "$LOG"
echo "LOG=$LOG"
