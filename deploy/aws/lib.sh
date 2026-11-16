#!/usr/bin/env bash
# Shared helpers for deploy/aws. Source this; don't run it.
#
# Instance access uses AWS Systems Manager (SSM) Run Command — the AWS counterpart of the GCP
# tooling's `gcloud compute ssh --tunnel-through-iap`: no SSH keys, no inbound ports, no public
# IPs. Files go through the project S3 bucket (instances read it with their instance role)
# instead of `gcloud compute scp`.

set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${AWS_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${AWS_DIR}/aws.env"

export AWS_REGION AWS_DEFAULT_REGION="${AWS_REGION}"
SCRIPT_NAME="$(basename "$0" .sh)"

log() { echo "[${SCRIPT_NAME}] $*" >&2; }
die() { echo "[${SCRIPT_NAME}] ERROR: $*" >&2; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed${2:+ ($2)}"; }
require_cmd aws "https://docs.aws.amazon.com/cli/"
require_cmd python3

# These scripts run on the operator's machine, which on macOS is bash 3.2: no `mapfile`, no
# negative array indexes, and "${empty[@]}" is an error under `set -u`. read_lines is the
# portable stand-in for `mapfile -t NAME < <(cmd)` (blank lines dropped).
read_lines() {  # read_lines <array-name>  < input
  local __line
  eval "$1=()"
  while IFS= read -r __line || [[ -n "${__line}" ]]; do
    [[ -n "${__line}" ]] && eval "$1+=(\"\${__line}\")"
  done
  return 0
}

require_count() {
  [[ "${1:-}" =~ ^[0-9]+$ && "$1" -gt 0 ]] || die "count must be a positive integer, got: '${1:-}'"
}

account_id() { aws sts get-caller-identity --query Account --output text; }
bucket_name() { echo "${BUCKET_BASE}-$(account_id)"; }

# --- Resource lookups (everything is tagged Project=$DEPLOY_PREFIX) ---------------------------------

project_tags() {  # tag spec for run-instances: project_tags <role> <name>
  echo "ResourceType=instance,Tags=[{Key=Project,Value=${DEPLOY_PREFIX}},{Key=Role,Value=$1},{Key=Name,Value=$2}]" \
       "ResourceType=volume,Tags=[{Key=Project,Value=${DEPLOY_PREFIX}},{Key=Role,Value=$1},{Key=Name,Value=$2}]"
}

lookup() {  # lookup <ec2 describe call...> -- prints the query result or empty
  aws ec2 "$@" --output text 2>/dev/null | grep -v '^None$' || true
}
ubuntu_ami() { aws ssm get-parameter --name "${UBUNTU_AMI_PARAM}" --query Parameter.Value --output text; }

# --- The Aerospike cluster (created with aerolab, outside these scripts) -----------------------
# AWS counterpart of deploy/gcp/load_runner.sh's discover_aerospike: aerolab tags every node with
# AEROLAB_CLUSTER_NAME (on GCP, the aerolab4cluster_name label). Sets AS_SEED_IP, AS_AZ, AS_VPC_ID,
# AS_SUBNET_ID and AS_SG_IDS (the cluster's security groups, which the fleet and app join so they
# can reach it) from the first running node.
discover_aerospike() {
  local line
  line="$(aws ec2 describe-instances \
    --filters "Name=tag:AEROLAB_CLUSTER_NAME,Values=${AEROSPIKE_CLUSTER_NAME}" "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].[Tags[?Key==`AEROLAB_NODE_NO`]|[0].Value,PrivateIpAddress,Placement.AvailabilityZone,VpcId,SubnetId,join(`,`,SecurityGroups[].GroupId)]' \
    --output text | sort -n | head -n1)"
  [[ -n "${line}" ]] || die "no running Aerospike cluster nodes tagged AEROLAB_CLUSTER_NAME=${AEROSPIKE_CLUSTER_NAME} in ${AWS_REGION} — create it with aerolab first (see README)"
  read -r _ AS_SEED_IP AS_AZ AS_VPC_ID AS_SUBNET_ID AS_SG_IDS <<<"${line}"
  AS_SG_IDS="${AS_SG_IDS//,/ }"
  log "Aerospike cluster '${AEROSPIKE_CLUSTER_NAME}': zone=${AS_AZ} seed=${AS_SEED_IP} vpc=${AS_VPC_ID}"
}

# --- Created on first use (GCP's create-demo-app-secure.sh likewise creates its service account
#     and firewall inline). All tagged Project=$DEPLOY_PREFIX in the cluster's VPC. ---------------
private_subnet_id() { lookup describe-subnets --filters "Name=tag:Project,Values=${DEPLOY_PREFIX}" "Name=tag:Tier,Values=private" --query 'Subnets[0].SubnetId'; }
security_group_id() { lookup describe-security-groups --filters "Name=tag:Project,Values=${DEPLOY_PREFIX}" "Name=group-name,Values=${DEPLOY_PREFIX}-sg" --query 'SecurityGroups[0].GroupId'; }
_tags() { echo "ResourceType=$1,Tags=[{Key=Project,Value=${DEPLOY_PREFIX}},{Key=Name,Value=$2}${3:+,$3}]"; }

# Private subnet in the cluster's zone (no public IPs — the AWS counterpart of GCP's --no-address),
# egress through a NAT gateway in the zone's default public subnet, a free S3 gateway endpoint, and
# a security group whose members may only talk to each other. Sets SUBNET_ID and SG_ID.
ensure_network() {
  discover_aerospike
  SUBNET_ID="$(private_subnet_id)"; SG_ID="$(security_group_id)"
  [[ -n "${SUBNET_ID}" && -n "${SG_ID}" ]] && return 0
  local vpc="${AS_VPC_ID}" public nat eip rt
  log "creating the private network in ${vpc} / ${AS_AZ} (once; NAT gateway ~\$0.045/hr while it exists)"
  if [[ -z "${SUBNET_ID}" ]]; then
    SUBNET_ID="$(aws ec2 create-subnet --vpc-id "${vpc}" --cidr-block "${PRIVATE_SUBNET_CIDR}" --availability-zone "${AS_AZ}" \
      --tag-specifications "$(_tags subnet "${DEPLOY_PREFIX}-private" "{Key=Tier,Value=private}")" --query Subnet.SubnetId --output text)"
  fi
  public="$(lookup describe-subnets --filters "Name=vpc-id,Values=${vpc}" "Name=availability-zone,Values=${AS_AZ}" \
    "Name=map-public-ip-on-launch,Values=true" --query 'Subnets[0].SubnetId')"
  [[ -n "${public}" ]] || die "no public subnet in ${AS_AZ} of ${vpc} to hold the NAT gateway"
  nat="$(lookup describe-nat-gateways --filter "Name=vpc-id,Values=${vpc}" "Name=tag:Project,Values=${DEPLOY_PREFIX}" \
    "Name=state,Values=pending,available" --query 'NatGateways[0].NatGatewayId')"
  if [[ -z "${nat}" ]]; then
    eip="$(aws ec2 allocate-address --domain vpc --tag-specifications "$(_tags elastic-ip "${DEPLOY_PREFIX}-nat-eip")" --query AllocationId --output text)"
    nat="$(aws ec2 create-nat-gateway --subnet-id "${public}" --allocation-id "${eip}" \
      --tag-specifications "$(_tags natgateway "${DEPLOY_PREFIX}-nat")" --query NatGateway.NatGatewayId --output text)"
    aws ec2 wait nat-gateway-available --nat-gateway-ids "${nat}"
  fi
  rt="$(lookup describe-route-tables --filters "Name=vpc-id,Values=${vpc}" "Name=tag:Project,Values=${DEPLOY_PREFIX}" --query 'RouteTables[0].RouteTableId')"
  if [[ -z "${rt}" ]]; then
    rt="$(aws ec2 create-route-table --vpc-id "${vpc}" --tag-specifications "$(_tags route-table "${DEPLOY_PREFIX}-private-rt")" \
      --query RouteTable.RouteTableId --output text)"
    aws ec2 associate-route-table --route-table-id "${rt}" --subnet-id "${SUBNET_ID}" >/dev/null
    aws ec2 create-route --route-table-id "${rt}" --destination-cidr-block 0.0.0.0/0 --nat-gateway-id "${nat}" >/dev/null
    aws ec2 create-vpc-endpoint --vpc-id "${vpc}" --service-name "com.amazonaws.${AWS_REGION}.s3" --route-table-ids "${rt}" \
      --tag-specifications "$(_tags vpc-endpoint "${DEPLOY_PREFIX}-s3")" >/dev/null
  fi
  if [[ -z "${SG_ID}" ]]; then
    SG_ID="$(aws ec2 create-security-group --group-name "${DEPLOY_PREFIX}-sg" --vpc-id "${vpc}" \
      --description "${DEPLOY_PREFIX}: fleet/app talk to each other; no inbound from outside" \
      --tag-specifications "$(_tags security-group "${DEPLOY_PREFIX}-sg")" --query GroupId --output text)"
    aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=${SG_ID}}]" >/dev/null
  fi
  log "private network ready: subnet ${SUBNET_ID}, security group ${SG_ID}, NAT ${nat}"
}

# Undo ensure_network once no instances use it (README "Cleanup"). Stops the NAT gateway's billing.
teardown_network() {
  local tagged="Name=tag:Project,Values=${DEPLOY_PREFIX}" x a
  for x in $(lookup describe-nat-gateways --filter "${tagged}" "Name=state,Values=pending,available" --query 'NatGateways[].NatGatewayId'); do
    aws ec2 delete-nat-gateway --nat-gateway-id "${x}" >/dev/null; aws ec2 wait nat-gateway-deleted --nat-gateway-ids "${x}"; log "deleted NAT ${x}"
  done
  for x in $(lookup describe-addresses --filters "${tagged}" --query 'Addresses[].AllocationId'); do aws ec2 release-address --allocation-id "${x}"; done
  for x in $(lookup describe-vpc-endpoints --filters "${tagged}" --query 'VpcEndpoints[].VpcEndpointId'); do aws ec2 delete-vpc-endpoints --vpc-endpoint-ids "${x}" >/dev/null; done
  for x in $(lookup describe-route-tables --filters "${tagged}" --query 'RouteTables[].RouteTableId'); do
    for a in $(lookup describe-route-tables --route-table-ids "${x}" --query 'RouteTables[0].Associations[].RouteTableAssociationId'); do
      aws ec2 disassociate-route-table --association-id "${a}"
    done
    aws ec2 delete-route-table --route-table-id "${x}"
  done
  for x in $(lookup describe-subnets --filters "${tagged}" --query 'Subnets[].SubnetId'); do aws ec2 delete-subnet --subnet-id "${x}"; done
  x="$(security_group_id)"; [[ -z "${x}" ]] || aws ec2 delete-security-group --group-id "${x}"
  log "network removed (bucket, secrets, roles and the aerolab cluster kept)"
}

# Private, encrypted bucket for artifacts and the dataset.
ensure_bucket() {
  local b; b="$(bucket_name)"
  aws s3api head-bucket --bucket "${b}" 2>/dev/null && return 0
  aws s3api create-bucket --bucket "${b}" --create-bucket-configuration "LocationConstraint=${AWS_REGION}" >/dev/null
  aws s3api put-public-access-block --bucket "${b}" \
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "${b}" \
    --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-tagging --bucket "${b}" --tagging "TagSet=[{Key=Project,Value=${DEPLOY_PREFIX}}]"
  log "created bucket s3://${b}"
}

# Least-privilege instance role + profile, kept in sync with the policy on every call.
#   fleet: SSM + read/write the project bucket (graph, loadgen, dataset generator)
#   app:   SSM + Bedrock invoke (Claude via us.* inference profiles) + demo-login secret + read the bucket
ensure_role() {  # ensure_role fleet|app   -> prints the instance profile name
  local kind="$1" role="${DEPLOY_PREFIX}-$1" acct b policy
  acct="$(account_id)"; b="$(bucket_name)"
  case "${kind}" in
    fleet) policy='{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":"arn:aws:s3:::'"${b}"'"},
 {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:AbortMultipartUpload"],"Resource":"arn:aws:s3:::'"${b}"'/*"}]}' ;;
    app) policy='{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],
  "Resource":["arn:aws:bedrock:*::foundation-model/anthropic.*","arn:aws:bedrock:*:'"${acct}"':inference-profile/us.anthropic.*"]},
 {"Effect":"Allow","Action":"secretsmanager:GetSecretValue","Resource":"arn:aws:secretsmanager:'"${AWS_REGION}:${acct}"':secret:'"${SECRET_DEMO_AUTH}"'-*"},
 {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":"arn:aws:s3:::'"${b}"'"},
 {"Effect":"Allow","Action":["s3:GetObject"],"Resource":"arn:aws:s3:::'"${b}"'/*"}]}' ;;
    *) die "unknown role kind: ${kind}" ;;
  esac
  if ! aws iam get-role --role-name "${role}" >/dev/null 2>&1; then
    aws iam create-role --role-name "${role}" --tags "Key=Project,Value=${DEPLOY_PREFIX}" \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    log "created role ${role}"
  fi
  aws iam attach-role-policy --role-name "${role}" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws iam put-role-policy --role-name "${role}" --policy-name "${role}-access" --policy-document "${policy}"
  if ! aws iam get-instance-profile --instance-profile-name "${role}" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "${role}" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "${role}" --role-name "${role}"
    sleep 10  # a new instance profile takes a moment before run-instances accepts it
  fi
  echo "${role}"
}

# Instance ids by role (running or pending), optionally a single name.
instance_ids() {  # instance_ids <role> [name]
  local filters=("Name=tag:Project,Values=${DEPLOY_PREFIX}" "Name=tag:Role,Values=$1"
                 "Name=instance-state-name,Values=pending,running,stopping,stopped")
  [[ -n "${2:-}" ]] && filters+=("Name=tag:Name,Values=$2")
  aws ec2 describe-instances --filters "${filters[@]}" \
    --query 'Reservations[].Instances[].InstanceId' --output text | tr '\t' '\n' | grep -v '^$' || true
}
instance_ip() { aws ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text; }
instance_by_name() { instance_ids "$1" "$2" | head -n1; }

# Launch instances: launch <role> <name> <type> <instance-profile> [extra run-instances args...]
launch() {
  local role="$1" name="$2" type="$3" profile="$4"; shift 4
  ensure_network
  # shellcheck disable=SC2046,SC2086
  aws ec2 run-instances \
    --image-id "$(ubuntu_ami)" \
    --instance-type "${type}" \
    --subnet-id "${SUBNET_ID}" \
    --security-group-ids "${SG_ID}" ${AS_SG_IDS} \
    --iam-instance-profile "Name=${profile}" \
    --no-associate-public-ip-address \
    --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled,InstanceMetadataTags=enabled" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${ROOT_VOLUME_GB},VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}" \
    --tag-specifications $(project_tags "${role}" "${name}") \
    "$@" \
    --query 'Instances[0].InstanceId' --output text
}

wait_running() { [[ $# -gt 0 ]] && aws ec2 wait instance-running --instance-ids "$@"; }

# Wait until every instance's SSM agent has registered (Online).
wait_ssm() {
  local ids=("$@") tries=60 online
  log "waiting for SSM agent on ${#ids[@]} instance(s)..."
  for ((t = 1; t <= tries; t++)); do
    online="$(aws ssm describe-instance-information \
      --filters "Key=InstanceIds,Values=$(IFS=,; echo "${ids[*]}")" \
      --query 'length(InstanceInformationList[?PingStatus==`Online`])' --output text 2>/dev/null || echo 0)"
    [[ "${online}" == "${#ids[@]}" ]] && { log "SSM online on all ${#ids[@]}"; return 0; }
    sleep 10
  done
  die "SSM agent did not come online on all instances (${online}/${#ids[@]}) — check NAT/egress and the instance role"
}

# Run a bash script on instances via SSM and print each instance's output.
# ssm_run <timeout-seconds> <script-text> <instance-id>...    (returns non-zero if any failed)
ssm_run() {
  local timeout="$1" script="$2"; shift 2
  local ids=("$@") params cmd_id rc=0 id status
  # Ship the script base64-encoded and run it with bash explicitly: AWS-RunShellScript doesn't
  # promise bash, and encoding sidesteps every quoting problem in the script text.
  params="$(python3 -c '
import base64, json, sys
b64 = base64.b64encode(sys.argv[1].encode()).decode()
cmd = ("f=$(mktemp); echo " + b64 + " | base64 -d > \"$f\"; bash \"$f\"; rc=$?; rm -f \"$f\"; exit $rc")
print(json.dumps({"commands": [cmd], "executionTimeout": [sys.argv[2]]}))' "${script}" "${timeout}")"
  cmd_id="$(aws ssm send-command --document-name AWS-RunShellScript --instance-ids "${ids[@]}" \
    --parameters "${params}" --timeout-seconds 600 --query Command.CommandId --output text)"
  for id in "${ids[@]}"; do
    while true; do
      status="$(aws ssm get-command-invocation --command-id "${cmd_id}" --instance-id "${id}" \
        --query Status --output text 2>/dev/null || echo Pending)"
      case "${status}" in Pending|InProgress|Delayed) sleep 3 ;; *) break ;; esac
    done
    aws ssm get-command-invocation --command-id "${cmd_id}" --instance-id "${id}" \
      --query '[StandardOutputContent,StandardErrorContent]' --output text | sed "s/^/[${id}] /"
    [[ "${status}" == "Success" ]] || { log "[${id}] command ${status}"; rc=1; }
  done
  return "${rc}"
}

# Same, but the script comes from a local file with `KEY=value` lines prepended as exports.
ssm_run_file() {  # ssm_run_file <timeout> <file> "<env exports>" <instance-id>...
  local timeout="$1" file="$2" envs="$3"; shift 3
  ssm_run "${timeout}" "$(printf '%s\n' "${envs}"; cat "${file}")" "$@"
}

# Upload a local file to the project bucket; prints the s3:// URI.
stage_file() {  # stage_file <local> <key>
  ensure_bucket
  local uri="s3://$(bucket_name)/$2"
  aws s3 cp --only-show-errors "$1" "${uri}"
  echo "${uri}"
}
