#!/usr/bin/env bash
# AWS port of deploy/gcp/create-demo-app-secure.sh: a hardened demo-app host.
#
# Compared to a plain instance, this:
#   - Has NO public IP (private subnet). Reach it only through SSM Session Manager port forwarding
#     — the AWS counterpart of the GCP IAP tunnel. No SSH keys, no inbound security-group rules.
#   - Runs with a dedicated, least-privilege instance role: Bedrock invoke for the Claude models,
#     read the demo-login secret, read the project bucket. No API keys on the box; no temporary
#     credentials that expire mid-demo.
#   - Requires IMDSv2 (hop limit 2 so the app's containers can use the instance role) and encrypts
#     the root volume.
#   - Ships the COMMITTED state of this repo (git archive HEAD) through the private bucket, so the
#     demo runs exactly what is on the branch, then builds and starts docker-compose.demo.yml.
#
# Usage:
#   ./create-demo-app.sh            # create (or redeploy onto) the app instance
#   ./create-demo-app.sh redeploy   # ship the current commit to the existing instance and rebuild
#   ./create-demo-app.sh destroy
source "$(dirname "$0")/lib.sh"

NAME="${DEPLOY_PREFIX}-app"

deploy() {
  local id="$1"
  local seed graph_id graph_ip tarball
  discover_aerospike; seed="${AS_SEED_IP}"
  graph_id="$(instance_by_name graph "${DEPLOY_PREFIX}-graph-1")"
  [[ -n "${graph_id}" ]] || die "no graph instance (${DEPLOY_PREFIX}-graph-1) — run ./load_runner.sh create 1 first"
  graph_ip="$(instance_ip "${graph_id}")"

  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] \
    || log "warning: uncommitted changes are NOT deployed — only HEAD ($(git -C "${REPO_ROOT}" rev-parse --short HEAD)) is shipped"
  local tgz; tgz="$(mktemp).tgz"
  git -C "${REPO_ROOT}" archive --format=tar.gz -o "${tgz}" HEAD
  tarball="$(stage_file "${tgz}" "artifacts/app-$(git -C "${REPO_ROOT}" rev-parse --short HEAD).tgz")"
  rm -f "${tgz}"

  log "deploying $(git -C "${REPO_ROOT}" rev-parse --short HEAD) to ${id} (aerospike seed ${seed}, graph ${graph_ip})"
  local envs="export AWS_REGION='${AWS_REGION}' APP_TARBALL='${tarball}' AEROSPIKE_HOST='${seed}' GRAPH_HOST_ADDRESS='${graph_ip}' SECRET_DEMO_AUTH='${SECRET_DEMO_AUTH}'"
  ssm_run_file 1800 "${AWS_DIR}/bootstrap-demo-app.sh" "${envs}" "${id}"
  instructions "${id}"
}

instructions() {
  cat <<EOF

== Demo app ${NAME} (${1}) has no public address. ==
Open the UI through SSM port forwarding (requires the Session Manager plugin:
https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html):

  # Frontend (demo UI) — then browse to http://localhost:8080
  aws ssm start-session --region ${AWS_REGION} --target ${1} \\
    --document-name AWS-StartPortForwardingSession --parameters portNumber=8080,localPortNumber=8080

  # Backend (the browser's investigation stream connects to localhost:4000)
  aws ssm start-session --region ${AWS_REGION} --target ${1} \\
    --document-name AWS-StartPortForwardingSession --parameters portNumber=4000,localPortNumber=4000

  # Shell on the box
  aws ssm start-session --region ${AWS_REGION} --target ${1}
EOF
}

case "${1:-create}" in
  create)
    id="$(instance_by_name app "${NAME}")"
    if [[ -z "${id}" ]]; then
      log "launching ${NAME} (${APP_INSTANCE_TYPE}) in the cluster zone"
      id="$(launch app "${NAME}" "${APP_INSTANCE_TYPE}" "$(ensure_role app)")"
      wait_running "${id}"; wait_ssm "${id}"
    else
      log "${NAME} exists (${id}); redeploying"
    fi
    deploy "${id}" ;;
  redeploy)
    id="$(instance_by_name app "${NAME}")"; [[ -n "${id}" ]] || die "${NAME} not found — run ./create-demo-app.sh"
    deploy "${id}" ;;
  destroy)
    id="$(instance_by_name app "${NAME}")"; [[ -n "${id}" ]] || { log "no app instance"; exit 0; }
    aws ec2 terminate-instances --instance-ids "${id}" >/dev/null
    aws ec2 wait instance-terminated --instance-ids "${id}"; log "destroyed ${id}" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
