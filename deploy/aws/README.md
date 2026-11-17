# AWS deployment: 5th-gen AMD EPYC + local NVMe

The AWS counterpart of `deploy/gcp/`, built for the re:Invent demo. As on GCP, **the Aerospike
cluster is created with aerolab, outside these scripts**; the scripts find it by its aerolab tag
(`AEROLAB_CLUSTER_NAME`, GCP: the `aerolab4cluster_name` label) and run the load fleet, demo app
and dataset pipeline next to it.

| Role | Instance (default) | Why |
|---|---|---|
| Aerospike cluster (aerolab) | **x8aedz.3xlarge** × 3 | The only 5th-gen AMD EPYC ("Turin") instance with **local NVMe**: 12 vCPU at 5.0 GHz, 384 GiB, 950 GB NVMe, 9.4 Gbps baseline (18.75 burst) |
| Graph + load-gen pairs | c8a.8xlarge | 5th-gen EPYC, 32 vCPU (the GCP fleet used n2d-standard-32) |
| Demo app | c8a.4xlarge | 5th-gen EPYC |
| Dataset generator | c5ad.24xlarge | Scratch NVMe for CSV generation only (2 × 1.9 TB); not part of the showcase |

x8aedz is offered in **us-west-2a/2b** (and eu-west-1), not us-east-1, so everything runs in
**us-west-2**. Settings live in `aws.env`; override any of them per command.

## What maps to what

| GCP | AWS |
|---|---|
| Aerospike cluster `demo-amd` built with aerolab (not in the repo) | Same: aerolab, cluster `demo-amd` — commands below |
| `create-demo-app-secure.sh` (no external IP, IAP, Secret Manager) | `create-demo-app.sh` — private subnet, **SSM Session Manager** port forwarding, Secrets Manager, **instance role for Bedrock** (no API keys, nothing expires mid-demo) |
| `bootstrap-demo-app.sh`, `fetch-secrets.sh`, `demo.env` | same names, AWS versions |
| `load_runner.sh`, `fleet-size-sweep.sh` | same interface; SSM instead of `gcloud compute ssh`, S3 instead of `scp`; one command fans out to the whole fleet |
| `run-loadgen.sh`, `loadgen.env`, `start-loadgen-50k.sh`, `bootstrap-loadgen.sh` | copied unchanged (not cloud-specific) |
| `bootstrap-ssd-raid.sh` | `bootstrap-nvme-raid.sh` (generator scratch) |
| `parallel-upload.sh`, `run-data-pipeline.sh` | S3 versions; the pipeline takes `<label>:<users>:<txns>` specs |
| `fix-csv-headers.sh` | `verify-csv-headers.sh` — the generator now writes headers on every shard, so it checks rather than repairs |
| inline `gcloud` in each script | `aws.env` (settings) + `lib.sh` (shared helpers). The private subnet + NAT, security group, bucket and instance roles are created on first use, the way `create-demo-app-secure.sh` creates its service account and firewall |

## Prerequisites

- AWS CLI v2 with credentials for the account, plus the
  [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
  to open the UI. `mvn` + JDK 21 locally to build the load generator. aerolab v8.
- The Aerospike Enterprise **feature-key file** (`features.conf`).
- Claude models enabled for the account in Amazon Bedrock (us.anthropic.* inference profiles).

## 1. Aerospike cluster (aerolab)

aerolab places nodes in the region's default VPC; us-west-2 has none in this account, so create it
once. Keep the feature key in Secrets Manager and hand aerolab a temporary copy.

```bash
aws ec2 create-default-vpc --region us-west-2                      # once per region
aws secretsmanager create-secret --region us-west-2 --name fraud-amd/aerospike-features-conf \
  --secret-string file://$HOME/path/features.conf                  # once: license -> Secrets Manager

aws secretsmanager get-secret-value --region us-west-2 --secret-id fraud-amd/aerospike-features-conf \
  --query SecretString --output text > /tmp/features.conf

aerolab config backend -t aws -r us-west-2
aerolab cluster create -n demo-amd -c 3 -v 8.0.0.7 -I x8aedz.3xlarge -U us-west-2a \
  -f /tmp/features.conf --start=n --aws-expire=0
rm -f /tmp/features.conf

# Raw NVMe partitions for the namespace (index stays in RAM, data on NVMe: HMA).
# 5 partitions using 85% of the drive leaves 15% overprovisioning, as in the Pegasus defaults.
aerolab cluster partition create -n demo-amd -t nvme -p 17,17,17,17,17
aerolab cluster partition conf -n demo-amd --namespace=test --filter-type=nvme --configure=device
aerolab conf adjust -n demo-amd set "namespace test.replication-factor" 2
aerolab conf adjust -n demo-amd set "namespace test.nsup-period" 120   # aerolab default is 0, which rejects every TTL write
aerolab aerospike start -n demo-amd
aerolab cluster list
```

`--aws-expire=0` disables aerolab's auto-expiry so the cluster can't vanish mid-event — destroy it
yourself afterwards with `aerolab cluster destroy -n demo-amd`. To use a different name, set
`AEROSPIKE_CLUSTER_NAME` for the scripts below.

## 2. Fleet and app

The first script run creates the private subnet (+ NAT gateway), security group, bucket and
instance roles in the cluster's VPC; later runs reuse them.

```bash
cd deploy/aws

# demo login -> Secrets Manager (once). The password goes through a private temp file, never argv.
read -r -s -p "demo password: " PW; echo
f=$(mktemp); DEMO_PW="$PW" python3 -c 'import json,os; print(json.dumps({"username":"demo","password":os.environ["DEMO_PW"]}))' > "$f"
aws secretsmanager create-secret --region us-west-2 --name fraud-amd/demo-auth --secret-string "file://$f"
rm -f "$f"; unset PW

./load_runner.sh create 4                           # 4 graph + 4 loadgen instances
./create-demo-app.sh                                # builds + starts the app from the committed branch

# throughput
KV_MODEL=flat BALANCES=false KV_TTL=600 ./load_runner.sh start 4 kv 0 300 64 1000000
./load_runner.sh report 4
KV_MODEL=flat BALANCES=false KV_TTL=600 MODE=kv ./fleet-size-sweep.sh 12 300   # find saturation
```

Open the UI with the two `aws ssm start-session ... AWS-StartPortForwardingSession` commands that
`create-demo-app.sh` prints, then browse to http://localhost:8080.

## 3. Dataset (optional, for remote / billion-scale mode)

As on GCP, the generator is a throwaway box: launch it, run the pipeline, terminate it.

```bash
bash <<'GEN'
source ./lib.sh
id=$(launch generator "${DEPLOY_PREFIX}-generator" "${GEN_INSTANCE_TYPE}" "$(ensure_role fleet)")
wait_running "$id"; wait_ssm "$id"
for f in run-data-pipeline.sh parallel-upload.sh verify-csv-headers.sh bootstrap-nvme-raid.sh; do stage_file "$f" "pipeline/$f"; done
stage_file ../../scripts/generate-ags-csv.py pipeline/generate-ags-csv.py
ssm_run 900 "set -e; snap install aws-cli --classic
mkdir -p /opt/pipeline; aws s3 cp --recursive --only-show-errors s3://$(bucket_name)/pipeline/ /opt/pipeline/
bash /opt/pipeline/bootstrap-nvme-raid.sh
nohup bash /opt/pipeline/run-data-pipeline.sh $(bucket_name) 1B:1000000000:0 >/dev/null 2>&1 &" "$id"
GEN

# progress; terminate once the log says the pipeline is done (it bills by the hour)
bash -c 'source ./lib.sh; ssm_run 60 "tail -n 20 /data/pipeline.log" $(instance_by_name generator ${DEPLOY_PREFIX}-generator)'
bash -c 'source ./lib.sh; aws ec2 terminate-instances --instance-ids $(instance_by_name generator ${DEPLOY_PREFIX}-generator)'
```

Bulk-load it with `emr-load.sh`, which runs the distributed Aerospike Graph bulk loader (Spark)
on an EMR cluster that terminates itself when the load ends. It needs the aerolab cluster up and
the loader jar staged once at `s3://<bucket>/loader/aerospike-graph-bulk-loader-3.3.3.jar`.

```bash
bash emr-load.sh 1B          # prints the EMR cluster id
aws emr list-steps --cluster-id <id> --query 'Steps[0].Status.State'   # RUNNING → COMPLETED
```

- **Size the cluster first.** The 1B dataset (3B vertices, 5B edges) takes ~7.0B records with
  RF 2: ~4.8 TB of NVMe and ~450 GB of index RAM. 5 × x8aedz.6xlarge holds it at ~59% NVMe fill.
- **Expect ~3 hours** with the default 8 × m8a.8xlarge workers (`LOADER_WORKERS`, `LOADER_TYPE`
  override them), at ~$18/hr for EMR.
- **Load into an empty namespace.** The loader refuses `-clear_existing_data` on an empty
  database, so the script doesn't pass it.
- **Progress** is in the step's driver log at
  `s3://<bucket>/loader/emr-logs/<cluster-id>/steps/<step-id>/stderr.gz` (uploaded every few
  minutes); it lists each loader phase and the vertex and edge totals when done.
- **Afterwards:** restart the Graph Service on the graph instances (`./load_runner.sh create <n>`
  re-runs their setup), and delete `s3://<bucket>/1B/loader-tmp/` (~485 GB of loader temp files).

## Sizing: set `KV_TTL` to the cluster

The load generator's max-throughput mode (`KV_MODEL=flat`) writes a new record per transaction
and expires it after `KV_TTL`. Records held = **writes/sec × TTL**, and each record costs 64
bytes of RAM per copy for the primary index (data is on NVMe):

```
RAM for the index ≈ writes/sec × KV_TTL × 64 B × replication factor
```

At 300k writes/s, TTL 600 s, RF 2: 180M records → ~23 GB of index RAM across the cluster, and at
~600 B per record ~216 GB of NVMe. x8aedz.3xlarge × 3 has 1.15 TB RAM and 2.85 TB NVMe, so the
binding limit there is NVMe at ~50% fill (defrag headroom), not RAM.

## Cost (us-west-2 on-demand, checked 2026-11-16)

| | $/hr |
|---|---|
| x8aedz.3xlarge × 3 (aerolab cluster) | 10.36 |
| c8a.8xlarge, per fleet instance (2 per pair) | ~1.72 |
| c8a.4xlarge app | 0.86 |
| NAT gateway | ~0.045 + data |
| c5ad.24xlarge generator (only while generating) | 4.13 |

## Cleanup

```bash
./load_runner.sh destroy 4
./create-demo-app.sh destroy
bash -c 'source ./lib.sh; teardown_network'        # NAT gateway, subnet, security group
aerolab cluster destroy -n demo-amd
```

The bucket (dataset), secrets and IAM roles are kept; delete them by hand if you mean to. Idle,
they cost little: the 1B CSVs (~950 GB) are ~$22/month in S3 Standard.

## Things to know

- **Instance-store NVMe is wiped if an instance stops** — aerolab nodes included.
- **The fleet and app join the cluster's security groups** (discovered from its nodes), so they can
  reach Aerospike without changing aerolab's rules.
- **Decision model:** `demo.env` sets `DECISION_MODEL=mock`. Strands Decider needs a serving
  endpoint on AWS (the strands-decider repo ships SageMaker and Lambda stacks); point
  `DECISION_MODEL=decider` + `DECISION_MODEL_URL` at it once deployed.
