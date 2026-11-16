# Run from deploy/aws:  bash emr-load.sh <label>      e.g. 1B
# Distributed Aerospike Graph bulk load (Spark on EMR) of s3://<bucket>/<label>/graph_csv.
source ./lib.sh
label="${1:?dataset label, e.g. 1B}"
ensure_network
b="$(bucket_name)"
props="$(mktemp)"
cat > "${props}" <<EOF
aerospike.client.host=${AS_SEED_IP}
aerospike.client.port=${AS_PORT}
aerospike.client.namespace=${AS_NAMESPACE}
aerospike.graphloader.vertices=s3://${b}/${label}/graph_csv/vertices
aerospike.graphloader.edges=s3://${b}/${label}/graph_csv/edges
aerospike.graphloader.temp-directory=s3://${b}/${label}/loader-tmp/
EOF
cat "${props}"
stage_file "${props}" "loader/${label}-bulk-loader.properties" >/dev/null; rm -f "${props}"

sgs="$(echo "${SG_ID} ${AS_SG_IDS}" | tr ' ' '\n' | grep . | sort -u | paste -sd, -)"
cid="$(aws emr create-cluster --name "${DEPLOY_PREFIX}-graph-loader-${label}" --release-label emr-7.12.0 \
  --applications Name=Spark --log-uri "s3://${b}/loader/emr-logs/" --use-default-roles \
  --ec2-attributes "SubnetId=${SUBNET_ID},AdditionalMasterSecurityGroups=[${sgs}],AdditionalSlaveSecurityGroups=[${sgs}]" \
  --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m8a.2xlarge \
                    InstanceGroupType=CORE,InstanceCount="${LOADER_WORKERS:-8}",InstanceType="${LOADER_TYPE:-m8a.8xlarge}" \
  --configurations '[{"Classification":"spark","Properties":{"maximizeResourceAllocation":"true"}}]' \
  --tags "Project=${DEPLOY_PREFIX}" --auto-terminate \
  --steps "Type=Spark,Name=graph-bulk-load-${label},ActionOnFailure=TERMINATE_CLUSTER,Args=[--class,com.aerospike.firefly.bulkloader.SparkBulkLoaderMain,s3://${b}/loader/aerospike-graph-bulk-loader-3.3.3.jar,-c,s3://${b}/loader/${label}-bulk-loader.properties]" \
  --query ClusterId --output text)"
log "EMR cluster ${cid} started; it terminates itself when the load step ends"
echo "${cid}"
