# vLLM + LMCache + Aerospike (KV-cache tier)

This is the demo-side plumbing for running the fraud investigation against a
self-hosted model with **Aerospike as the KV-cache tier behind LMCache**, and for
measuring whether that tier can pay off at all.

Two independent pieces, in the order they should be done:

| | What | Needs a GPU? |
|---|---|---|
| **0** | [Measure prefix reuse](#0-measure-prefix-reuse-first) — how much prompt text repeats across investigations | No |
| **1** | [Point the agent at vLLM](#1-point-the-agent-at-vllm-phase-1) instead of the hosted model | Yes |
| **2** | [Put Aerospike behind LMCache](#2-put-aerospike-behind-lmcache-phase-2) and measure cold vs warm | Yes |

Step 0 decides whether 1 and 2 are worth running: **prefix caching only matches from
the start of the prompt**, so if investigations do not share long prefixes, no KV tier
helps and the result is a flat line regardless of how fast the storage is.

---

## 0. Measure prefix reuse first

```bash
python backend/benchmarks/prefix_reuse.py                      # both layouts, 16-token blocks
python backend/benchmarks/prefix_reuse.py --block-tokens 16,256 --investigations 50
python backend/benchmarks/prefix_reuse.py --layouts current --json reuse-current.json
```

No GPU, no LLM spend, no Aerospike, no dependencies beyond the standard library. It
renders the **real** prompts — the production templates in `workflow/specialists.py`,
the production evidence builder, and the block assembly in `workflow/prompt_layout.py`
— for a synthetic corpus of investigations, then chain-hashes each prompt into
fixed-size token blocks and counts the leading run a warm cache would already hold.

Block sizes worth reporting: **16** tokens (vLLM's page size, so this is what vLLM's
own prefix cache sees) and **256** tokens (LMCache's default chunk, so this is what
lands in Aerospike).

### What it found here, and the layout switch

The ReAct prompt is assembled from five blocks. The historical order puts the
per-iteration `## STATUS` counter *immediately before* the append-only
`## CONVERSATION`, so iteration N+1 shares no prefix with iteration N past that point
and **the entire conversation is re-prefilled every iteration**. The specialist prompts
also carry the case evidence twice — once from the specialist template, once from the
ReAct assembly.

`PROMPT_LAYOUT` selects the order:

| Layout | Block order | Notes |
|---|---|---|
| `current` (default) | system → evidence → tools → **status** → conversation | byte-identical to what the engine has always sent |
| `stable_first` | system → tools → evidence → conversation → **status** | mutating block last; duplicate evidence dropped |

Measured on 20 investigations × 4 nodes = 580 prompts
(`--investigations 20 --block-tokens 16,256`):

```
layout          block  prompts   blocks   reused   block %     tokens   token %
current            16      580    95992    42228     44.0%    1530968     44.1%
current           256      580     6273     2382     38.0%    1530968     39.8%
stable_first       16      580    85159    69215     81.3%    1357651     81.6%
stable_first      256      580     5586     4078     73.0%    1357651     76.9%

@ 16-token blocks (vLLM pages):   +37.3 pts reuse, recomputed 855,320 -> 250,211
@ 256-token blocks (LMCache):     +35.0 pts reuse, recomputed 921,176 -> 313,683
tokens sent (both):               1,530,968 -> 1,357,651  (-11.3%)
```

Reuse is lower at 256-token blocks than at 16, as expected: a coarser block wastes
more of the tail of the matching region. The per-node breakdown shows the network
analyst gaining most (39.2% → 86.7% at 16 tokens) because it runs the longest tool
loop, so it has the most history being needlessly re-prefilled.

The token reduction is the duplicate-evidence fix; the reuse jump is the reorder.
Both are measured with `--seed` fixed, so the two layouts see identical cases.

**The default is still `current`.** Flipping it changes what the model sees, and block
reordering can shift model behaviour even when the content is identical — so treat
`PROMPT_LAYOUT=stable_first` as a change to validate on report quality first, then land
as the default. Note also that vLLM does automatic prefix caching on its own: this gain
is real *before* any external KV tier exists, which is why it should be measured (and
credited) separately from Aerospike.

---

## 1. Point the agent at vLLM (Phase 1)

`LLM_PROVIDER=vllm` sends both engines to any OpenAI-compatible `/v1` endpoint.

```bash
LLM_PROVIDER=vllm \
VLLM_BASE_URL=http://localhost:8000/v1 \
VLLM_MODEL=Qwen/Qwen3-8B \
INVESTIGATION_ENGINE=langgraph   # or adk
```

- **LangGraph engine** — `workflow/nodes/langgraph_agent.py` posts the ReAct prompt to
  `/chat/completions` as a single user message, so the served engine sees one stable
  byte stream per iteration.
- **ADK engine** — `resolve_adk_model` returns a `LiteLlm` bound to `openai/<model>`
  with `api_base` set, so ADK routes through LiteLLM to the same endpoint.
- Startup logs a reachability check (`/v1/models`) and names the served model ids if
  the configured one is missing. A failed check is non-fatal — investigations fall back
  to the deterministic assessment.

Keep the prompt **identical** to the hosted-model run when comparing, and change one
variable at a time.

### Running vLLM from this repo

Stock vLLM and LMCache images do **not** include the Aerospike connector, so build an
image first:

```bash
docker build -f vllm.Dockerfile -t lmcache-aerospike/vllm-openai:rocm .

# NVIDIA host instead of ROCm:
docker build -f vllm.Dockerfile --build-arg BASE_IMAGE=vllm/vllm-openai:latest \
    -t lmcache-aerospike/vllm-openai:cuda .
```

The Dockerfile tries the published `lmcache-aerospike` wheel first and verifies the
connector is importable, so a broken image fails the build rather than the demo. If no
wheel matches the platform, rebuild LMCache from source with the connector compiled in
— it needs the Aerospike **C client** present:

```bash
docker build -f vllm.Dockerfile --build-arg BUILD_FROM_SOURCE=1 \
    -t lmcache-aerospike/vllm-openai:rocm .
```

Then start it (it is behind a compose profile, so it never runs by accident — and it
needs `/dev/kfd` + `/dev/dri`, i.e. an actual AMD GPU host):

```bash
VLLM_IMAGE=lmcache-aerospike/vllm-openai:rocm docker compose --profile vllm up vllm
```

---

## 2. Put Aerospike behind LMCache (Phase 2)

`config/lmcache.yaml` configures the tiering: L1 GPU → L2 pinned host DRAM →
remote Aerospike. It is mounted into the container at `/etc/lmcache/lmcache.yaml`
(`LMCACHE_CONFIG_FILE`).

Connector settings, and the environment variables that override them:

| Parameter | Default | Notes |
|---|---|---|
| `hosts` | required | `host:port[,host:port…]` seed list (`LMCACHE_AEROSPIKE_HOSTS`) |
| `namespace` | `lmcache` | must exist, **with `nsup-period > 0`** (`LMCACHE_AEROSPIKE_NAMESPACE`) |
| `set_name` | `kv_chunks` | `LMCACHE_AEROSPIKE_SET` |
| `num_workers` | 8 | C++ I/O threads — the real I/O queue depth; raise for throughput |
| `read_timeout_ms` / `write_timeout_ms` | 1000 / 2000 | |
| `default_ttl_seconds` | 86400 | 0 falls back to the namespace `default-ttl` |
| `target_segment_bytes` | 0 | 0 = auto-discover the server record cap and shard to fit |
| `max_record_bytes` | 0 | override the detected cap |
| `max_capacity_gb` | 0 | > 0 enables client-side capacity tracking, hence eviction |
| `username` / `password` | "" | Enterprise (`LMCACHE_AEROSPIKE_USERNAME` / `_PASSWORD`) |

### The namespace (already in `config/aerospike.conf`)

```
namespace lmcache {
    replication-factor 1
    default-ttl 3600
    nsup-period 60          # MUST be > 0
    storage-engine device { file /opt/aerospike/data/lmcache.dat  filesize 4G }
}
```

**Applying it needs a container recreate, and that costs the seeded data.** The
`aerospike/aerospike-server-*` entrypoint renders
`/etc/aerospike/aerospike.template.conf` over `/etc/aerospike/aerospike.conf` on every
fresh container, so a config bind-mounted at that path is overwritten before `asd`
starts — silently, and on the host side too. `docker-compose.yaml` therefore mounts
this repo's config at `/opt/aerospike/etc/aerospike.conf` read-only and starts the
server with `--config-file`, which is what makes the file authoritative. Consequence:

```bash
docker compose up -d --force-recreate aerospike-db     # picks up the lmcache namespace
docker exec aerospike-db asinfo -v namespaces          # expect: test;lmcache
```

The `test` namespace has no data volume, so a recreate is a cold start — reseed
afterwards (wipe → bulk-load → inject transactions → detect).

Three things to keep in mind:

- **`nsup-period 0` is the classic trap.** Nothing is ever reaped, the namespace fills,
  `stop-writes` trips, and it presents as an Aerospike capacity fault rather than a
  config error.
- **Separate namespace from `test` on purpose.** KV chunks are multi-megabyte and churn
  on TTL; the fraud records are small and never expire. Co-locating them makes the two
  profiles fight over defrag and device bandwidth.
- **Records are 8 MB max, and copy-on-write rewrites the whole record.** The connector
  stores a meta record plus payload segments and shards at the discovered cap, so a
  default 256-token chunk of a dense GQA model (~320 KB/token at fp16 → ~80 MB) becomes
  ~10 segments. Quantizing KV to fp8/MXFP4 or cutting `chunk_size` reduces that
  amplification directly.

### What to measure

Cold versus warm on repeated investigations of the same account, then across accounts:

- **TTFT p50 / p99**, cold vs warm — the user-visible number
- **Prefix hit rate; tokens reused vs recomputed** — explains *why* TTFT moved
- **Goodput per GPU at a fixed TTFT target** — the metric AMD uses for Infera
- **Concurrent investigations at a fixed SLO** — the same win expressed as capacity
- **Feature-lookup p99** — already collected by this demo; include it so the CPU and
  GPU halves of the data layer are reported on one page

Sanity check before believing any of it: confirm reuse is actually high (step 0), and
confirm `asadm -e "info namespace lmcache"` shows objects arriving and expiring.

---

## Files

| Path | Purpose |
|---|---|
| `backend/workflow/prompt_layout.py` | prompt blocks + `PROMPT_LAYOUT` order |
| `backend/benchmarks/prefix_reuse.py` | prefix-reuse measurement (no GPU) |
| `backend/workflow/llm.py` | `LLM_PROVIDER=vllm` config and ADK/LangChain wiring |
| `backend/workflow/health.py` | `/v1/models` reachability check at startup |
| `vllm.Dockerfile` | vLLM + LMCache + Aerospike connector image |
| `config/lmcache.yaml` | LMCache tiering + connector settings |
| `config/aerospike.conf` | the `lmcache` namespace |
| `docker-compose.yaml` | `vllm` profile (ROCm devices, LMCache env) |
