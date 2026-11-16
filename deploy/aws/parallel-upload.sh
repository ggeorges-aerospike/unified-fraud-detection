#!/usr/bin/env bash
# AWS port of deploy/gcp/parallel-upload.sh: highly parallel, resumable upload of a graph_csv tree
# to S3. Re-running is safe: `aws s3 sync` skips objects already uploaded with the same size.
# Runs on the generator host with its instance role (no keys); traffic uses the VPC's S3 endpoint.
#
# Usage: ./parallel-upload.sh <src-dir> <s3-prefix>
#   e.g. ./parallel-upload.sh /data/graph_csv s3://fraud-amd-data-<account>/1B
set -euo pipefail

SRC="${1:?source dir, e.g. /data/graph_csv}"
DEST="${2:?s3 prefix, e.g. s3://fraud-amd-data-<account>/1B}"
command -v aws >/dev/null || sudo snap install aws-cli --classic

# Many concurrent part uploads; large parts for multi-GB CSV shards.
aws configure set default.s3.max_concurrent_requests "${S3_CONCURRENCY:-64}"
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.max_queue_size 10000

DEST_PATH="${DEST%/}/graph_csv"
LOG="/data/upload-$(echo "${DEST}" | tr '/:' '--').log"
echo "Uploading ${SRC} -> ${DEST_PATH}/"
echo "Started: $(date -Is)"
aws s3 sync "${SRC}" "${DEST_PATH}" --only-show-errors 2>&1 | tee -a "${LOG}"
echo "Done: $(date -Is)"
aws s3 ls --recursive --summarize --human-readable "${DEST_PATH}/" | tail -2
