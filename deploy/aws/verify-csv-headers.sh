#!/usr/bin/env bash
# Replaces deploy/gcp/fix-csv-headers.sh on AWS.
#
# The GCP script repaired an old generator bug (headers written only on shard 0) by prepending a
# header to every part file with GCS server-side `compose`. The generator now writes a header on
# EVERY shard (scripts/generate-ags-csv.py, write_headers = True), so freshly generated data needs
# no repair — and S3 has no cheap in-place prepend. This script CHECKS instead: it reads the first
# byte of every CSV under the prefix (the Aerospike Graph bulk-load header starts with "~") and
# fails, listing the files, if any is missing its header. Fix those by regenerating.
#
# Usage: ./verify-csv-headers.sh s3://bucket/1B/graph_csv
set -euo pipefail
PREFIX_URI="${1:?s3 prefix, e.g. s3://bucket/1B/graph_csv}"
PARALLEL="${PARALLEL:-32}"
BUCKET="$(echo "${PREFIX_URI}" | sed -E 's#^s3://([^/]+)/?.*#\1#')"
KEY_PREFIX="$(echo "${PREFIX_URI}" | sed -E 's#^s3://[^/]+/?##')"

check_one() {
  local first
  first="$(aws s3api get-object --bucket "$1" --key "$2" --range bytes=0-0 /dev/stdout 2>/dev/null | head -c1 | tail -c1)"
  [[ "${first}" == "~" ]] || echo "MISSING_HEADER s3://$1/$2"
}
export -f check_one

total="$(aws s3api list-objects-v2 --bucket "${BUCKET}" --prefix "${KEY_PREFIX}" --query 'Contents[?ends_with(Key, `.csv`)].Key' --output text | tr '\t' '\n' | grep -c . || true)"
missing="$(aws s3api list-objects-v2 --bucket "${BUCKET}" --prefix "${KEY_PREFIX}" --query 'Contents[?ends_with(Key, `.csv`)].Key' --output text \
  | tr '\t' '\n' | grep . | xargs -P "${PARALLEL}" -I{} bash -c 'check_one "$0" "$1"' "${BUCKET}" {} || true)"

if [[ -n "${missing}" ]]; then
  echo "${missing}"
  echo "FAIL: $(echo "${missing}" | grep -c .) of ${total} CSV file(s) under ${PREFIX_URI} lack the bulk-load header"
  exit 1
fi
echo "OK: all ${total} CSV file(s) under ${PREFIX_URI} start with a bulk-load header"
