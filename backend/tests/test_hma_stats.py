"""HMA statistics: parsing Aerospike info output, cluster-wide summing, and tier grouping."""

from services.hma_stats import (parse_kv, parse_latencies, parse_records, summarize, summarize_throughput,
                                tier_of)

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


LAT_A = ("batch-index:msec,0.0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00;"
         "{bar}-write:;{test}-read:msec,1000.0,2.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00;"
         "{test}-write:msec,233623.1,0.93,0.40,0.10,0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00;"
         "{test}-udf:")
LAT_B = "{test}-write:msec,100000.0,0.10,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00;{test}-read:"


def test_parse_latencies_reads_only_this_namespace():
    lat = parse_latencies(LAT_A, "test")
    assert set(lat) == {"read", "write"}  # idle and other-namespace histograms are skipped
    assert lat["write"]["ops_per_sec"] == 233623.1
    assert lat["write"]["over_ms"][1] == 0.93 and lat["write"]["over_ms"][8] == 0.02
    assert parse_latencies("", "test") == {}


def test_summarize_throughput_sums_nodes_and_weights_latency():
    nodes = [
        {"node": "B", "latencies": parse_latencies(LAT_B, "test"),
         "sets": parse_records("ns=test:set=transactions_flat:objects=400:data_used_bytes=4000;ns=test:set=users:objects=9")},
        {"node": "A", "latencies": parse_latencies(LAT_A, "test"),
         "sets": parse_records("ns=test:set=transactions_flat:objects=600:data_used_bytes=6000")},
    ]
    t = summarize_throughput(nodes, replication_factor=2)
    assert t["writes_per_sec"] == 333623.1
    assert t["reads_per_sec"] == 1000.0
    # Weighted by ops: (233623.1*0.93 + 100000*0.10) / 333623.1
    assert t["write_over_ms"][1] == round((233623.1 * 0.93 + 100000 * 0.10) / 333623.1, 2)
    assert t["read_over_ms"][1] == 2.0
    assert [n["node"] for n in t["nodes"]] == ["A", "B"]
    # Stream records are de-replicated; index RAM counts every copy.
    assert t["stream"] == {"sets": ["transactions_flat"], "records": 500, "index_bytes": 1000 * 64, "data_bytes": 10000}
    assert tier_of("transactions_flat") == "stream"


def test_summarize_throughput_idle_cluster():
    t = summarize_throughput([{"node": "A", "latencies": {}, "sets": []}], replication_factor=0)
    assert t["writes_per_sec"] == 0 and t["write_over_ms"][1] is None and t["replication_factor"] == 1
