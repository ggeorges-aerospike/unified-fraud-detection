"""HMA statistics: parsing Aerospike info output, cluster-wide summing, and tier grouping."""

from services.hma_stats import parse_kv, parse_records, summarize, tier_of

NODE_A = parse_kv("objects=1000;master_objects=500;index_used_bytes=64000;sindex_used_bytes=33554432;"
                  "set_index_used_bytes=0;data_used_bytes=600000;data_total_bytes=4294967296;data_avail_pct=97;"
                  "data_compression_ratio=1.000")
NODE_B = parse_kv("objects=1000;master_objects=500;index_used_bytes=64000;sindex_used_bytes=33554432;"
                  "set_index_used_bytes=0;data_used_bytes=600000;data_total_bytes=4294967296;data_avail_pct=97")
SETS = parse_records("ns=test:set=users:objects=400:data_used_bytes=300000:sindexes=1;"
                     "ns=test:set=triage_log:objects=100:data_used_bytes=100000:sindexes=1;"
                     "ns=test:set=0_2:objects=300:data_used_bytes=150000:sindexes=0;"
                     "ns=test:set=strands_sessions:objects=200:data_used_bytes=50000:sindexes=0;"
                     "ns=test:set=mystery:objects=0:data_used_bytes=0:sindexes=0")
SINDEXES = [
    {"indexname": "idx_user_wf_status", "set": "users", "bin": "wf_status", "type": "string",
     "entries": "10", "used_bytes": "16777216"},
    {"indexname": "idx_triage_user_id", "set": "triage_log", "bin": "user_id", "type": "string",
     "entries": "100", "used_bytes": "16777216"},
]
CONFIG = parse_kv("index-type=shmem;sindex-type=shmem;storage-engine=device;replication-factor=2;"
                  "storage-engine.device[0]=/dev/nvme1n1;storage-engine.device[1]=/dev/nvme2n1;"
                  "storage-engine.compression=none;storage-engine.read-page-cache=true")


def test_parsers():
    assert parse_kv("a=1;b=x=y;;") == {"a": "1", "b": "x=y"}
    [rec] = parse_records("ns=test:set=a:objects=3;")
    assert rec == {"ns": "test", "set": "a", "objects": "3"}


def test_tier_mapping():
    assert tier_of("0_2") == "graph" and tier_of("FIREFLYWARMUP_9") == "graph"
    assert tier_of("transactions") == "predictive"
    assert tier_of("triage_log") == "decision"
    assert tier_of("strands_memory") == "agents" and tier_of("adk_sessions") == "agents"
    assert tier_of("lg_cp_w") == "agents" and tier_of("investigations") == "agents"
    assert tier_of("mystery") == "other"


def test_two_node_cluster_sums_and_groups():
    d = summarize("test", [NODE_A, NODE_B], [SETS, SETS], [SINDEXES, SINDEXES], CONFIG,
                  build="8.0.0.7", edition="Aerospike Enterprise Edition", node_count=2)
    assert d["nodes"] == 2 and d["records"] == 1000 and d["objects_all_copies"] == 2000
    assert d["ram"]["primary_index_bytes"] == 128000
    assert d["ram"]["secondary_index_bytes"] == 2 * 33554432
    assert d["ram"]["secondary_index_count"] == 2
    assert d["ram"]["secondary_index_uniform_allocation"] == 2 * 16777216     # summed across nodes, still uniform
    assert d["per_record"] == {"index_bytes": 64.0, "data_bytes": 600.0}
    assert d["ssd"]["kind"] == "raw device" and d["ssd"]["devices"] == ["/dev/nvme1n1", "/dev/nvme2n1"]
    assert d["config"]["replication_factor"] == 2
    tiers = {t["id"]: t for t in d["tiers"]}
    assert set(tiers) == {"graph", "predictive", "decision", "agents", "other"}
    assert tiers["predictive"]["objects"] == 800 and tiers["predictive"]["index_bytes"] == 800 * 64
    assert tiers["decision"]["data_bytes"] == 200000 and tiers["decision"]["sindexes"] == 1
    assert tiers["agents"]["sets"] == ["strands_sessions"]
    assert tiers["other"]["sets"] == []                                         # empty sets are not listed


def test_file_backed_and_non_uniform_sindexes():
    ix = [dict(SINDEXES[0]), dict(SINDEXES[1], used_bytes="33554432")]
    cfg = parse_kv("storage-engine=device;storage-engine.file[0]=/opt/aerospike/data/test.dat")
    d = summarize("test", [NODE_A], [SETS], [ix], cfg)
    assert d["ssd"]["kind"] == "file" and d["ssd"]["devices"] == ["/opt/aerospike/data/test.dat"]
    assert d["ram"]["secondary_index_uniform_allocation"] is None


def test_empty_namespace_has_no_division_errors():
    d = summarize("test", [parse_kv("objects=0;index_used_bytes=0;data_used_bytes=0")], [[]], [[]], {})
    assert d["records"] == 0 and d["per_record"] == {"index_bytes": None, "data_bytes": None}
    assert d["tiers"] == []
