#!/usr/bin/env bash
# AWS port of deploy/gcp/fetch-secrets.sh: materializes .env from deploy/aws/demo.env plus the demo
# login from Secrets Manager, using the instance role (no long-lived keys). Run on the demo-app host.
#
# Usage: ./fetch-secrets.sh [env-file]     (default: <repo>/.env)
# Needs: SECRET_DEMO_AUTH (default fraud-amd/demo-auth) and AWS_REGION in the environment.
set -euo pipefail

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${1:-${DIR}/.env}"
TEMPLATE="${DIR}/deploy/aws/demo.env"
SECRET="${SECRET_DEMO_AUTH:-fraud-amd/demo-auth}"
REGION="${AWS_REGION:-us-west-2}"

AUTH_JSON="$(aws secretsmanager get-secret-value --region "${REGION}" --secret-id "${SECRET}" \
  --query SecretString --output text 2>/dev/null || echo '{}')"
AUTH_USER="$(jq -r '.username // empty' <<<"${AUTH_JSON}")"
AUTH_PASS="$(jq -r '.password // empty' <<<"${AUTH_JSON}")"
[[ -n "${AUTH_USER}" ]] || echo "warning: no demo login in secret '${SECRET}' — the UI will be open (no sign-in)" >&2

umask 077
AUTH_USER="${AUTH_USER}" AUTH_PASS="${AUTH_PASS}" python3 - "${TEMPLATE}" "${ENV_FILE}" <<'EOF'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
values = {"AUTH_USERNAME": os.environ.get("AUTH_USER", ""), "AUTH_PASSWORD": os.environ.get("AUTH_PASS", "")}
out = []
for line in open(src):
    key = line.split("=", 1)[0]
    out.append(f"{key}={values[key]}\n" if key in values else line)
open(dst, "w").writelines(out)
EOF
chmod 600 "${ENV_FILE}"
echo "Wrote ${ENV_FILE} (mode 600) from deploy/aws/demo.env + Secrets Manager."
