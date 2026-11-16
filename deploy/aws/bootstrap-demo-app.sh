# shellcheck shell=bash
# Runs ON the demo app instance as root (sent by create-demo-app.sh through SSM), with exported:
#   AWS_REGION  APP_TARBALL (s3:// URI of the packaged repo)  AEROSPIKE_HOST  GRAPH_HOST_ADDRESS
#   SECRET_DEMO_AUTH  DATA_SOURCE_MODE  AGS_IMAGE  AS_NAMESPACE
# AWS counterpart of deploy/gcp/bootstrap-demo-app.sh + fetch-secrets.sh + `docker compose up`.
set -euo pipefail
exec > >(tee -a /var/log/demo-app-bootstrap.log) 2>&1
echo "[demo-app-bootstrap] $(date -Is) starting"
export DEBIAN_FRONTEND=noninteractive

# OS patching stays on (as on GCP): a box months behind on fixes is a much worse day.
if ! dpkg -s unattended-upgrades >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq unattended-upgrades jq
  cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF
  systemctl enable --now unattended-upgrades
fi
command -v jq >/dev/null || apt-get install -y -qq jq
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
command -v aws >/dev/null || snap install aws-cli --classic

APP_DIR=/opt/unified-fraud-detection
mkdir -p "${APP_DIR}"
aws s3 cp --only-show-errors "${APP_TARBALL}" /tmp/app.tgz
tar -xzf /tmp/app.tgz -C "${APP_DIR}" && rm -f /tmp/app.tgz

# local mode: Graph Service beside the app, reading the same data dir the backend writes the CSVs to.
mkdir -p "${APP_DIR}/data"
if [[ "${DATA_SOURCE_MODE}" == local ]]; then
  GRAPH_HOST_ADDRESS="$(hostname -I | awk '{print $1}')"
  if ! docker ps -a --format '{{.Names}}' | grep -qx ags; then
    docker run -d --name ags --restart unless-stopped -p 8182:8182 -v "${APP_DIR}/data:/data" \
      -e aerospike.client.host="${AEROSPIKE_HOST}" -e aerospike.client.namespace="${AS_NAMESPACE}" "${AGS_IMAGE}"
  fi
fi

# .env from the template + cluster addresses + the demo login from Secrets Manager.
"${APP_DIR}/deploy/aws/fetch-secrets.sh" "${APP_DIR}/.env"
sed -i -e "s#^AEROSPIKE_HOST=.*#AEROSPIKE_HOST=${AEROSPIKE_HOST}#" \
       -e "s#^GRAPH_HOST_ADDRESS=.*#GRAPH_HOST_ADDRESS=${GRAPH_HOST_ADDRESS}#" \
       -e "s#^DATA_SOURCE_MODE=.*#DATA_SOURCE_MODE=${DATA_SOURCE_MODE}#" \
       -e "s#^AWS_REGION=.*#AWS_REGION=${AWS_REGION}#" "${APP_DIR}/.env"

cd "${APP_DIR}"
docker compose -f docker-compose.demo.yml --env-file .env up -d --build
docker compose -f docker-compose.demo.yml ps
echo "[demo-app-bootstrap] done"
