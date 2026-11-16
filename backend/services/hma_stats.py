"""Hybrid Memory Architecture (HMA) statistics: what lives in RAM vs on SSD, from the server itself.

Aerospike keeps every record's primary-index entry (64 bytes) in RAM and the record data on
flash. This module reads the server's own info statistics (summed across nodes), groups the
namespace's sets by the fraud-funnel tier that owns them, and reports per-record averages that a
UI can project to larger scales. Nothing here is estimated except what is labelled as such:
secondary-index memory is reported as allocated, including its fixed per-index reservation.

Parsing is pure (``parse_*`` / ``summarize``) so it is testable without a server.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

PRIMARY_INDEX_BYTES_PER_RECORD = 64  # documented Aerospike primary-index entry size

# Which funnel tier owns which set. Unknown sets fall into "other".
TIERS: List[Dict[str, Any]] = [
    {"id": "graph", "label": "Graph (Aerospike Graph)",
     "match": lambda s: bool(re.fullmatch(r"\d+_\d+", s)) or s.startswith("FIREFLYWARMUP")},
    {"id": "predictive", "label": "Predictive model & core data",
     "match": lambda s: s in {"users", "transactions", "account_fact", "device_fact", "flagged_accounts",
                              "flagged_queue", "fraud_feed", "evaluations", "detection_history", "config",
                              "workflow"}},
    {"id": "decision", "label": "Decision model",
     "match": lambda s: s in {"triage_log"}},
    {"id": "agents", "label": "AI agents (state & memory)",
     "match": lambda s: s in {"investigations", "case_memory"} or s.startswith(("strands_", "adk_", "lg_cp"))},
]


def parse_kv(text: str, sep: str = ";") -> Dict[str, str]:
    """'a=1;b=2' -> {'a': '1', 'b': '2'} (tolerates empty items and values containing '=')."""
    out: Dict[str, str] = {}
    for item in (text or "").strip().split(sep):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_records(text: str) -> List[Dict[str, str]]:
    """'ns=test:set=a:objects=1;ns=test:set=b:objects=2' -> [{...}, {...}]."""
    return [parse_kv(item, ":") for item in (text or "").strip().split(";") if item.strip()]


def _num(v: Optional[str]) -> int:
    try:
        return int(float(v)) if v not in (None, "") else 0
    except ValueError:
        return 0


def tier_of(set_name: str) -> str:
    for t in TIERS:
        if t["match"](set_name):
            return t["id"]
    return "other"


def summarize(
    namespace: str,
    node_ns_stats: Iterable[Dict[str, str]],
    node_sets: Iterable[List[Dict[str, str]]],
    node_sindexes: Iterable[List[Dict[str, Any]]],
    config: Dict[str, str],
    build: str = "",
    edition: str = "",
    node_count: int = 1,
) -> Dict[str, Any]:
    """Combine per-node info output into one cluster-wide HMA summary."""
    ns_stats = list(node_ns_stats)
    total = lambda key: sum(_num(s.get(key)) for s in ns_stats)  # noqa: E731

    master_objects = total("master_objects") or total("objects")
    objects_all_copies = total("objects")
    index_bytes = total("index_used_bytes")
    sindex_bytes = total("sindex_used_bytes")
    set_index_bytes = total("set_index_used_bytes")
    data_bytes = total("data_used_bytes")
    data_total = total("data_total_bytes")

    # Per-set, summed across nodes (objects include replicas on multi-node clusters).
    sets: Dict[str, Dict[str, Any]] = {}
    for node in node_sets:
        for rec in node:
            name = rec.get("set")
            if not name:
                continue
            s = sets.setdefault(name, {"set": name, "objects": 0, "data_bytes": 0, "sindexes": 0})
            s["objects"] += _num(rec.get("objects"))
            s["data_bytes"] += _num(rec.get("data_used_bytes"))
            s["sindexes"] = max(s["sindexes"], _num(rec.get("sindexes")))

    # Secondary indexes: per index, memory as allocated (summed across nodes).
    sindexes: Dict[str, Dict[str, Any]] = {}
    for node in node_sindexes:
        for ix in node:
            name = ix.get("indexname")
            if not name:
                continue
            d = sindexes.setdefault(name, {"name": name, "set": ix.get("set"), "bin": ix.get("bin"),
                                           "type": ix.get("type"), "entries": 0, "used_bytes": 0})
            d["entries"] += _num(ix.get("entries"))
            d["used_bytes"] += _num(ix.get("used_bytes"))

    tiers = []
    for t in TIERS + [{"id": "other", "label": "Other"}]:
        members = [s for s in sets.values() if tier_of(s["set"]) == t["id"]]
        if not members:
            continue
        objs = sum(s["objects"] for s in members)
        tiers.append({
            "id": t["id"], "label": t["label"],
            "sets": sorted((s["set"] for s in members if s["objects"]), key=str),
            "objects": objs,
            "index_bytes": objs * PRIMARY_INDEX_BYTES_PER_RECORD,
            "data_bytes": sum(s["data_bytes"] for s in members),
            "sindexes": sum(1 for ix in sindexes.values() if tier_of(ix["set"] or "") == t["id"]),
            "sindex_bytes": sum(ix["used_bytes"] for ix in sindexes.values() if tier_of(ix["set"] or "") == t["id"]),
        })

    devices = [v for k, v in sorted(config.items()) if re.fullmatch(r"storage-engine\.(file|device)\[\d+\]", k)]
    sindex_list = sorted(sindexes.values(), key=lambda d: (-d["used_bytes"], d["name"]))
    sizes = {ix["used_bytes"] for ix in sindex_list if ix["used_bytes"]}
    return {
        "namespace": namespace,
        "nodes": node_count,
        "build": build,
        "edition": edition,
        "records": master_objects,
        "objects_all_copies": objects_all_copies,
        "ram": {
            "primary_index_bytes": index_bytes,
            "secondary_index_bytes": sindex_bytes,
            "set_index_bytes": set_index_bytes,
            "total_bytes": index_bytes + sindex_bytes + set_index_bytes,
            "secondary_index_count": len(sindex_list),
            # When every index reports the same allocation, it is a fixed reservation, not usage.
            "secondary_index_uniform_allocation": next(iter(sizes)) if len(sizes) == 1 else None,
        },
        "ssd": {
            "data_bytes": data_bytes,
            "capacity_bytes": data_total,
            "used_pct": round(100 * data_bytes / data_total, 2) if data_total else None,
            "avail_pct": _num(ns_stats[0].get("data_avail_pct")) if ns_stats else None,
            "devices": devices,
            "kind": "raw device" if any(k.startswith("storage-engine.device[") for k in config) else
                    "file" if devices else config.get("storage-engine", ""),
        },
        "per_record": {
            "index_bytes": round(index_bytes / objects_all_copies, 1) if objects_all_copies else None,
            "data_bytes": round(data_bytes / objects_all_copies, 1) if objects_all_copies else None,
        },
        "config": {
            "index_type": config.get("index-type"),
            "sindex_type": config.get("sindex-type"),
            "storage_engine": config.get("storage-engine"),
            "replication_factor": _num(config.get("replication-factor")) or None,
            "compression": config.get("storage-engine.compression"),
            "read_page_cache": config.get("storage-engine.read-page-cache"),
            "data_compression_ratio": (ns_stats[0].get("data_compression_ratio") if ns_stats else None),
        },
        "tiers": tiers,
        "sindexes": sindex_list,
    }


def collect(client: Any, namespace: str) -> Dict[str, Any]:
    """Query every node of a connected Aerospike client and summarize."""

    def per_node(cmd: str) -> List[str]:
        out = []
        for _node, (err, resp) in (client.info_all(cmd) or {}).items():
            if err is None and resp is not None:
                out.append(resp.strip().split("\t", 1)[-1] if "\t" in resp else resp.strip())
        return out

    ns_stats = [parse_kv(r) for r in per_node(f"namespace/{namespace}")]
    sets = [[r for r in parse_records(t) if r.get("ns", namespace) == namespace] for t in per_node(f"sets/{namespace}")]

    sindexes: List[List[Dict[str, Any]]] = []
    lists = per_node(f"sindex-list:ns={namespace}")
    if lists:
        names = [r for r in parse_records(lists[0]) if r.get("indexname")]
        node_stats: Dict[int, List[Dict[str, Any]]] = {}
        for ix in names:
            for i, stat in enumerate(per_node(f"sindex/{namespace}/{ix['indexname']}")):
                node_stats.setdefault(i, []).append({**ix, **parse_kv(stat)})
        sindexes = list(node_stats.values())

    config = parse_kv((per_node(f"get-config:context=namespace;id={namespace}") or [""])[0])
    build = (per_node("build") or [""])[0]
    edition = (per_node("edition") or [""])[0]
    return summarize(namespace, ns_stats, sets, sindexes, config, build, edition, node_count=len(ns_stats))
