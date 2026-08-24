# vLLM (ROCm) + LMCache with the Aerospike remote connector.
#
# Stock vLLM and LMCache images do NOT carry the Aerospike backend, so the
# `vllm` compose profile needs an image built here first:
#
#   docker build -f vllm.Dockerfile -t lmcache-aerospike/vllm-openai:rocm .
#
# On an NVIDIA box, override the base:
#   docker build -f vllm.Dockerfile --build-arg BASE_IMAGE=vllm/vllm-openai:latest \
#       -t lmcache-aerospike/vllm-openai:cuda .
#
# Two install paths, in order of preference:
#   1. the published `lmcache-aerospike` wheel (no compilation);
#   2. building LMCache from source with BUILD_AEROSPIKE=1, which needs the
#      Aerospike C client headers present (see BUILD_FROM_SOURCE below).
ARG BASE_IMAGE=rocm/vllm:latest
FROM ${BASE_IMAGE}

ARG LMCACHE_VERSION=
ARG BUILD_FROM_SOURCE=0
ARG AEROSPIKE_C_CLIENT_VERSION=6.6.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Path 1 — wheels. Kept as its own layer so a failure here is obvious in the log.
RUN pip install --no-cache-dir \
      "lmcache${LMCACHE_VERSION:+==${LMCACHE_VERSION}}" \
      lmcache-aerospike

# Path 2 — source build of the native connector. Only runs when asked for, since
# it pulls a toolchain and the Aerospike C client into the image.
RUN if [ "${BUILD_FROM_SOURCE}" = "1" ]; then \
      set -euo pipefail; \
      apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libssl-dev zlib1g-dev; \
      git clone --depth 1 --recurse-submodules \
        https://github.com/aerospike/aerospike-client-c.git /tmp/as-c; \
      make -C /tmp/as-c && make -C /tmp/as-c install; \
      pip uninstall -y lmcache || true; \
      BUILD_AEROSPIKE=1 AEROSPIKE_INCLUDE_DIR=/usr/local/include \
        pip install --no-cache-dir --no-binary :all: \
        "lmcache${LMCACHE_VERSION:+==${LMCACHE_VERSION}}"; \
      rm -rf /tmp/as-c /var/lib/apt/lists/*; \
    fi

# Fail the build rather than the demo if the connector is not importable.
RUN python -c "import lmcache, importlib.util as u; \
assert u.find_spec('lmcache_aerospike') or u.find_spec('lmcache.v1.storage_backend.connector.aerospike_connector'), \
'LMCache is installed but the Aerospike connector is not — rebuild with --build-arg BUILD_FROM_SOURCE=1'; \
print('LMCache + Aerospike connector present')"

ENV LMCACHE_CONFIG_FILE=/etc/lmcache/lmcache.yaml
COPY config/lmcache.yaml /etc/lmcache/lmcache.yaml

ENTRYPOINT ["vllm", "serve"]
