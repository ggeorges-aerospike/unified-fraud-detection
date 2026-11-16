#!/usr/bin/env bash
# AWS port of deploy/gcp/load_runner.sh: a scale-out fleet of paired Aerospike Graph Service +
# load-generator instances, so throughput is multiplied by running N independent
# (graph, loadgen) pairs against the Aerospike cluster.
#
# Usage (same interface as the GCP script):
#   ./load_runner.sh create  <count>                      # idempotent: tops the fleet up to <count> pairs
#   ./load_runner.sh destroy <count>
#   ./load_runner.sh start   <count> [loadgen-args...]   # forwarded to run-loadgen.sh
#   ./load_runner.sh stop    <count> [loadgen|graph|all]  # default: all
#   ./load_runner.sh status  <count>
#   ./load_runner.sh report  <count>                      # achieved txn/s across the fleet
#
# `stop` never stops or terminates instances (only `destroy` does) — it kills the loadgen java
# process and/or stops the `ags` graph container, so a stopped fleet stays warm for a restart.
#
# Examples:
#   ./load_runner.sh create 10
#   ./load_runner.sh start 10 paired 50000 3600 64 0
#   KV_MODEL=flat BALANCES=false KV_TTL=600 ./load_runner.sh start 20 kv 0 300 64 0   # max Aerospike throughput
#   ./load_runner.sh report 10
#   ./load_runner.sh destroy 10
#
# The Aerospike cluster is found by its aerolab tag (AEROSPIKE_CLUSTER_NAME, default demo-amd —
# GCP's discover_aerospike). Instances are named <prefix>-graph-<i> / <prefix>-loadgen-<i>, launched
# in the cluster's zone on the private subnet (no public IPs; managed through SSM, not SSH). Each
# operation is ONE SSM command fanned out to the whole fleet; a loadgen finds its pair number
# from its own Name tag (instance-metadata tags) instead of one SSH session per instance.
source "$(dirname "$0")/lib.sh"

AEROSPIKE_NAMESPACE="${AS_NAMESPACE}"
graph_name() { echo "${DEPLOY_PREFIX}-graph-$1"; }
loadgen_name() { echo "${DEPLOY_PREFIX}-loadgen-$1"; }

usage() { sed -n '2,25p' "$0"; exit 1; }


# Running instance ids of a role for pairs 1..n (missing ones are skipped with a warning).
fleet_ids() {  # fleet_ids <graph|loadgen> <n>
  local role="$1" n="$2" i id name
  for i in $(seq 1 "${n}"); do
    name="$( [[ ${role} == graph ]] && graph_name "${i}" || loadgen_name "${i}")"
    id="$(instance_by_name "${role}" "${name}")"
    if [[ -n "${id}" ]]; then echo "${id}"; else log "warning: ${name} not found"; fi
  done
}

# --- create ----------------------------------------------------------------------------------
GRAPH_SETUP='set -euo pipefail
exec > >(tee -a /var/log/graph-bootstrap.log) 2>&1
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
docker pull "${AGS_IMAGE}"
if docker ps --format "{{.Names}}" | grep -qx ags; then echo "ags already running"
elif docker ps -a --format "{{.Names}}" | grep -qx ags; then docker start ags
else
  docker run -d --name ags --restart unless-stopped -p '"${GRAPH_PORT}"':8182 \
    -e aerospike.client.host="${AS_SEED_IP}" \
    -e aerospike.client.namespace="${AEROSPIKE_NAMESPACE}" \
    "${AGS_IMAGE}"
fi
echo "ags up (seed=${AS_SEED_IP} ns=${AEROSPIKE_NAMESPACE})"'

LOADGEN_SETUP='set -euo pipefail
exec > >(tee -a /var/log/loadgen-bootstrap.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
command -v java >/dev/null || { apt-get update -qq && apt-get install -y -qq openjdk-21-jre-headless curl; }
command -v aws >/dev/null || snap install aws-cli --classic
mkdir -p /opt/fraud-loadgen
echo "loadgen host ready: $(java -version 2>&1 | head -1)"'

cmd_create() {
  local n="$1"; require_count "${n}"
  ensure_network
  local seed="${AS_SEED_IP}" profile   # seed set by ensure_network -> discover_aerospike
  profile="$(ensure_role fleet)"
  local new_graph=() new_loadgen=() i
  for i in $(seq 1 "${n}"); do
    if [[ -z "$(instance_by_name graph "$(graph_name "${i}")")" ]]; then
      new_graph+=("$(launch graph "$(graph_name "${i}")" "${FLEET_INSTANCE_TYPE}" "${profile}")")
    else log "skip $(graph_name "${i}") (already exists)"; fi
    if [[ -z "$(instance_by_name loadgen "$(loadgen_name "${i}")")" ]]; then
      new_loadgen+=("$(launch loadgen "$(loadgen_name "${i}")" "${FLEET_INSTANCE_TYPE}" "${profile}")")
    else log "skip $(loadgen_name "${i}") (already exists)"; fi
  done
  local all=(${new_graph[@]+"${new_graph[@]}"} ${new_loadgen[@]+"${new_loadgen[@]}"})
  if [[ ${#all[@]} -eq 0 ]]; then log "fleet already has ${n} pair(s)"; cmd_status; return 0; fi
  log "launched ${#new_graph[@]} graph + ${#new_loadgen[@]} loadgen x ${FLEET_INSTANCE_TYPE}"
  wait_running "${all[@]}"; wait_ssm "${all[@]}"
  local envs="export AGS_IMAGE='${AGS_IMAGE}' AS_SEED_IP='${seed}' AEROSPIKE_NAMESPACE='${AEROSPIKE_NAMESPACE}'"
  [[ ${#new_graph[@]} -gt 0 ]] && ssm_run 900 "${envs}"$'\n'"${GRAPH_SETUP}" "${new_graph[@]}"
  [[ ${#new_loadgen[@]} -gt 0 ]] && ssm_run 900 "${LOADGEN_SETUP}" "${new_loadgen[@]}"
  log "create complete"
  cmd_status
}

# --- destroy -------------------------------------------------------------------------------
cmd_destroy() {
  local n="$1"; require_count "${n}"
  read_lines ids < <(fleet_ids graph "${n}"; fleet_ids loadgen "${n}")
  [[ ${#ids[@]} -gt 0 ]] || { log "nothing to destroy"; return 0; }
  log "terminating ${#ids[@]} fleet instance(s)"
  aws ec2 terminate-instances --instance-ids "${ids[@]}" >/dev/null
  aws ec2 wait instance-terminated --instance-ids "${ids[@]}"
  log "destroy complete"
}

# --- start ---------------------------------------------------------------------------------
cmd_start() {
  local n="$1"; shift; require_count "${n}"
  local loadgen_args=("$@")
  [[ ${#loadgen_args[@]} -gt 0 ]] || loadgen_args=(paired 50000 3600 64 0)
  discover_aerospike
  local seed="${AS_SEED_IP}"

  if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    require_cmd mvn "needed to build the loadgen jar; or set SKIP_BUILD=1 to reuse loadgen/target/fraud-loadgen.jar"
    log "building loadgen jar (mvn package; set SKIP_BUILD=1 to reuse the existing jar)"
    (cd "${REPO_ROOT}/loadgen" && mvn -q -DskipTests package)
  fi
  local jar="${REPO_ROOT}/loadgen/target/fraud-loadgen.jar"
  [[ -f "${jar}" ]] || die "jar not found at ${jar} (build failed?)"
  local jar_uri run_uri
  jar_uri="$(stage_file "${jar}" artifacts/fraud-loadgen.jar)"
  run_uri="$(stage_file "${AWS_DIR}/run-loadgen.sh" artifacts/run-loadgen.sh)"

  # Pair i's graph IP, as "i=ip" entries the loadgens pick from by their own index.
  local graph_map="" i gid
  for i in $(seq 1 "${n}"); do
    gid="$(instance_by_name graph "$(graph_name "${i}")")"
    [[ -n "${gid}" ]] || die "$(graph_name "${i}") not found — run ./load_runner.sh create ${n}"
    graph_map+="${i}=$(instance_ip "${gid}") "
  done
  read_lines lg_ids < <(fleet_ids loadgen "${n}")
  [[ ${#lg_ids[@]} -eq "${n}" ]] || die "expected ${n} loadgen instance(s), found ${#lg_ids[@]}"

  local exports="" v
  for v in FRAUD_RATIO MULES FRAUDSTERS RING_POOL_SIZE RING_RATIO ACCOUNT_PREFIX USER_PREFIX \
           KV_MODEL KV_MAX_CONNS BALANCES KV_TTL READ_RATIO; do
    [[ -n "${!v:-}" ]] && exports+="export ${v}='${!v}'"$'\n'
  done

  log "starting ${n} loadgen(s) against their paired graph node, Aerospike seed=${seed}"
  log "loadgen args: ${loadgen_args[*]}"
  local script
  script="$(cat <<EOF
set -euo pipefail
cd /opt/fraud-loadgen
aws s3 cp --only-show-errors '${jar_uri}' fraud-loadgen.jar
aws s3 cp --only-show-errors '${run_uri}' run-loadgen.sh && chmod +x run-loadgen.sh
TOKEN="\$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')"
NAME="\$(curl -sf -H "X-aws-ec2-metadata-token: \${TOKEN}" http://169.254.169.254/latest/meta-data/tags/instance/Name)"
IDX="\${NAME##*-}"
GRAPH_IP=""
for kv in ${graph_map}; do [[ "\${kv%%=*}" == "\${IDX}" ]] && GRAPH_IP="\${kv#*=}"; done
[[ -n "\${GRAPH_IP}" ]] || { echo "no graph IP for pair \${IDX}"; exit 1; }
cat > loadgen.env <<ENVEOF
AS_HOST=${seed}
AS_PORT=${AS_PORT}
AS_NAMESPACE=${AEROSPIKE_NAMESPACE}
GRAPH_HOST=\${GRAPH_IP}
GRAPH_PORT=${GRAPH_PORT}
ENVEOF
pkill -x java 2>/dev/null || true; sleep 2
${exports}
# Stagger so the fleet's connection pools don't hit the cluster in one burst.
sleep \$(( (IDX - 1) * ${START_STAGGER_SECS:-2} ))
nohup ./run-loadgen.sh $(printf '%q ' "${loadgen_args[@]}") > loadgen.log 2>&1 < /dev/null &
sleep 1; echo "started pair \${IDX} -> graph \${GRAPH_IP}"
EOF
)"
  ssm_run 900 "${script}" "${lg_ids[@]}" || die "one or more loadgens failed to start (see above)"
  log "start complete: ${n} loadgen instance(s) running"
}

# --- stop ----------------------------------------------------------------------------------
cmd_stop() {
  local n="$1" target="${2:-all}"; require_count "${n}"
  case "${target}" in loadgen|graph|all) ;; *) die "stop target must be loadgen, graph or all" ;; esac
  if [[ "${target}" != graph ]]; then
    read_lines ids < <(fleet_ids loadgen "${n}")
    [[ ${#ids[@]} -gt 0 ]] && ssm_run 120 'pkill -x java 2>/dev/null || true; echo stopped' "${ids[@]}" || true
  fi
  if [[ "${target}" != loadgen ]]; then
    read_lines ids < <(fleet_ids graph "${n}")
    [[ ${#ids[@]} -gt 0 ]] && ssm_run 120 'docker stop ags 2>/dev/null || true; echo stopped' "${ids[@]}" || true
  fi
  log "stop complete (instances still running)"
}

# --- status --------------------------------------------------------------------------------
cmd_status() {
  aws ec2 describe-instances --filters "Name=tag:Project,Values=${DEPLOY_PREFIX}" "Name=tag:Role,Values=graph,loadgen" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,InstanceType,State.Name,PrivateIpAddress,InstanceId]' \
    --output text | sort | column -t
}

# --- report --------------------------------------------------------------------------------
# Metrics.java logs "mode=... throughput=N txn/s total=N errors=N" once per second per instance.
cmd_report() {
  local n="$1"; require_count "${n}"
  read_lines ids < <(fleet_ids loadgen "${n}")
  [[ ${#ids[@]} -gt 0 ]] || die "no loadgen instances"
  log "sampling latest per-instance throughput from ${#ids[@]} loadgen log(s)..."
  local out
  out="$(ssm_run 60 'f=/opt/fraud-loadgen/loadgen.log
TOKEN="$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")"
NAME="$(curl -sf -H "X-aws-ec2-metadata-token: ${TOKEN}" http://169.254.169.254/latest/meta-data/tags/instance/Name)"
p=$(grep -oE "throughput=[0-9]+" "$f" 2>/dev/null | cut -d= -f2 | sort -rn | head -1)
echo "REPORT ${NAME} peak_tps=${p:-0} $(grep "c.a.f.Metrics" "$f" 2>/dev/null | tail -1)"' "${ids[@]}" || true)"

  local total_tps=0 total_peak=0 total_txns=0 total_errors=0 up=0 line name tps peak txns errs
  printf "%-26s %12s %12s %14s %10s\n" "INSTANCE" "TXN/S" "PEAK_TXN/S" "TOTAL_TXNS" "ERRORS"
  while read -r line; do
    name="$(awk '{print $3}' <<<"${line}")"
    peak="$(grep -oE 'peak_tps=[0-9]+' <<<"${line}" | cut -d= -f2 || true)"
    tps="$(grep -oE 'throughput=[0-9]+' <<<"${line}" | cut -d= -f2 || true)"
    txns="$(grep -oE 'total=[0-9]+' <<<"${line}" | cut -d= -f2 || true)"
    errs="$(grep -oE 'errors=[0-9]+' <<<"${line}" | cut -d= -f2 || true)"
    if [[ -z "${tps}" ]]; then printf "%-26s %12s %12s %14s %10s\n" "${name}" "no data" - - -; continue; fi
    printf "%-26s %12s %12s %14s %10s\n" "${name}" "${tps}" "${peak:-0}" "${txns:-0}" "${errs:-0}"
    total_tps=$((total_tps + tps)); total_peak=$((total_peak + ${peak:-0}))
    total_txns=$((total_txns + ${txns:-0})); total_errors=$((total_errors + ${errs:-0})); up=$((up + 1))
  done < <(grep ' REPORT ' <<<"${out}" | sort -k3,3)
  echo "---"
  log "fleet achieved: ${total_tps} txn/s (sum of peak seconds: ${total_peak}) across ${up}/${n} reporting instance(s), ${total_txns} total txns this run, ${total_errors} errors"
}

[[ $# -ge 1 ]] || usage
COMMAND="$1"; shift || true
[[ $# -ge 1 ]] || usage
case "${COMMAND}" in
  create)  cmd_create "$@" ;;
  destroy) cmd_destroy "$@" ;;
  start)   cmd_start "$@" ;;
  stop)    cmd_stop "$@" ;;
  status)  cmd_status ;;
  report)  cmd_report "$1" ;;
  *) usage ;;
esac
