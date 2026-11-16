#!/usr/bin/env bash
# AWS port of deploy/gcp/run-data-pipeline.sh, parameterized instead of hard-coded 1B -> 10B.
# Runs on the generator host (see README: Dataset). For each dataset label:
# generate the graph CSVs onto /data, upload them to S3, verify every CSV has its header,
# then delete the local copy to free scratch for the next one.
#
# Usage: ./run-data-pipeline.sh <bucket> <label>:<users>:<txns-per-user> [<label>:<users>:<txns>...]
#   e.g. ./run-data-pipeline.sh fraud-amd-data-123456789012 1B:1000000000:0 10B:10000000000:2
set -euo pipefail
BUCKET="${1:?bucket name}"; shift
[[ $# -ge 1 ]] || { echo "need at least one <label>:<users>:<txns-per-user>"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=/data/pipeline.log
exec > >(tee -a "$LOG") 2>&1
WORKERS="${WORKERS:-$(nproc)}"

echo "=== Pipeline start $(date -Is) (workers=${WORKERS}) ==="
for spec in "$@"; do
  IFS=: read -r label users txns <<<"${spec}"
  [[ -n "${label}" && -n "${users}" && -n "${txns}" ]] || { echo "bad spec '${spec}'"; exit 1; }
  echo "--- ${label}: generate ${users} users (${txns} txns/user) ---"
  rm -rf /data/graph_csv
  python3 "${HERE}/generate-ags-csv.py" --users "${users}" --txns-per-user "${txns}" \
    --out_dir /data/graph_csv --workers "${WORKERS}" --seed 42 | tee "/data/generate-${label}.log"
  df -h /data

  echo "--- ${label}: upload to s3://${BUCKET}/${label}/graph_csv ---"
  bash "${HERE}/parallel-upload.sh" /data/graph_csv "s3://${BUCKET}/${label}"

  echo "--- ${label}: verify headers ---"
  bash "${HERE}/verify-csv-headers.sh" "s3://${BUCKET}/${label}/graph_csv"

  echo "--- ${label}: remove local copy ---"
  rm -rf /data/graph_csv
done
echo "=== Pipeline complete $(date -Is) ==="
aws s3 ls "s3://${BUCKET}/"
