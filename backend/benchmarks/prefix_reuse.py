#!/usr/bin/env python3
"""
Prefix-reuse measurement for the fraud-investigation prompts.

A KV-cache tier — vLLM's own prefix cache, LMCache's L2/L3, Aerospike behind
LMCache — pays off only to the extent that prompt text repeats *from the start of
the prompt*. On workloads with little shared prefix the whole thing produces a flat
line no matter how fast the storage is. So this is the measurement that decides
whether the rest of the POC is worth running, and it needs no GPU, no LLM spend and
no Aerospike.

What it does: renders the **real** ReAct prompts (the production templates in
``workflow/specialists.py`` and the block assembly in ``workflow/prompt_layout.py``,
with evidence from the production ``build_evidence_summary``) for a synthetic corpus
of investigations, chunks each prompt into fixed-size token blocks, chain-hashes the
blocks the way a serving engine does, and reports how many blocks a warm cache would
already hold.

Because the hash of block *k* covers blocks 0..k, a single changed byte early in the
prompt invalidates everything after it — which is exactly why block order matters.
Run it against both layouts to see the cost of the current order:

    python backend/benchmarks/prefix_reuse.py
    python backend/benchmarks/prefix_reuse.py --layouts current --json out.json
    python backend/benchmarks/prefix_reuse.py --block-tokens 16,256 --investigations 50

Caveat on tokens: block boundaries are estimated at ``--chars-per-token`` characters
(default 4.0) unless ``--tokenizer`` names an installed HF tokenizer. The estimate is
fine for *comparing* layouts, which is what this harness is for; it is not a
substitute for the served engine's own hit-rate counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.assessment import build_evidence_summary  # noqa: E402
from workflow.prompt_layout import (  # noqa: E402
    LAYOUTS,
    default_hint,
    history_block,
    iter_layouts,
    render_react_prompt,
    status_lines,
    tool_lines,
)
from workflow.specialists import (  # noqa: E402
    INVESTIGATOR_SYSTEM,
    SPECIALIST_NAMES,
    _SPECIALIST_SPECS,
    _SPECIALIST_SYSTEM,
    _SPECIALIST_TOOLS,
)
from workflow.tools.investigation_tools import InvestigationTools  # noqa: E402

INVESTIGATOR_TOOLS = [
    "get_account_transactions",
    "get_counterparty_profile",
    "get_counterparty_transactions",
    "get_account_risk_features",
    "get_device_risk_features",
    "detect_fraud_ring",
    "get_transaction_network",
]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic case corpus — different accounts, same policy/instructions, which is
# the shape of the real workload (one flagged account per investigation).
# ─────────────────────────────────────────────────────────────────────────────
_TYPOLOGIES = ("money_mule", "account_takeover", "fraud_ring", "card_testing", "promo_abuse")
_CITIES = ("NY", "SF", "Austin", "Miami", "Chicago", "Seattle", "Boston", "Denver")
_JOBS = ("Engineer", "Teacher", "Driver", "Nurse", "Contractor", "Analyst", "Chef")


def _case(rng: random.Random, idx: int) -> Tuple[Dict, Dict]:
    """Build one (initial_evidence, alert_evidence) pair, shaped like the real KV reads."""
    user_id = f"U{2_000_000 + idx * 7919:07d}"
    n_accounts = rng.randint(1, 3)
    n_devices = rng.randint(1, 3)

    accounts, account_facts = {}, {}
    for a in range(n_accounts):
        acct = f"A{idx:04d}{a:06d}"
        accounts[acct] = {
            "type": rng.choice(("checking", "savings", "credit")),
            "balance": round(rng.uniform(50, 90_000), 2),
            "is_fraud": rng.random() < 0.4,
            "status": rng.choice(("active", "active", "restricted")),
            "opened_date": f"202{rng.randint(0, 5)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        }
        account_facts[acct] = {
            "txn_count_7d": rng.randint(0, 400),
            "out_amount_7d": round(rng.uniform(0, 250_000), 2),
            "avg_out_amount_7d": round(rng.uniform(10, 5_000), 2),
            "distinct_counterparties_7d": rng.randint(0, 60),
            "amount_zscore_7d": round(rng.uniform(-2, 9), 2),
            "night_txn_ratio_7d": round(rng.random(), 3),
            "risk_score": rng.randint(0, 100),
        }

    devices, device_facts = {}, {}
    for d in range(n_devices):
        dev = f"D{idx:04d}{d:04d}"
        devices[dev] = {"os": rng.choice(("iOS", "Android", "Windows")), "trusted": rng.random() < 0.6}
        device_facts[dev] = {
            "accounts_seen": rng.randint(1, 12),
            "users_seen": rng.randint(1, 9),
            "is_shared": rng.random() < 0.3,
            "risk_score": rng.randint(0, 100),
        }

    initial = {
        "user_id": user_id,
        "profile": {
            "name": f"Case {idx} Holder",
            "location": rng.choice(_CITIES),
            "occupation": rng.choice(_JOBS),
            "account_age_days": rng.randint(3, 3_000),
            "risk_score": rng.randint(30, 99),
            "kyc_status": rng.choice(("complete", "pending")),
        },
        "accounts": accounts,
        "devices": devices,
        "account_facts": account_facts,
        "device_facts": device_facts,
        "account_metrics": {
            "total_transactions": rng.randint(10, 5_000),
            "flagged_transactions": rng.randint(0, 90),
        },
    }
    alert = {
        "trigger_type": rng.choice(("ML", "RULE")),
        "trigger_rule": rng.choice(("Pattern Anomaly Detection", "Velocity Burst", "Shared Device Ring")),
        "original_score": rng.randint(60, 99),
        "flag_reason": f"Suspected {rng.choice(_TYPOLOGIES)}",
        "trigger_timestamp": f"2026-08-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00",
        "previous_flags_count": rng.randint(0, 4),
    }
    return initial, alert


def _tool_result_text(rng: random.Random, tool: str, size: int) -> str:
    """A tool result of realistic size. Content is arbitrary but deterministic."""
    rows = []
    while sum(len(r) for r in rows) < size:
        rows.append(
            f'{{"id": "T{rng.randint(10**6, 10**7)}", "amount": {round(rng.uniform(5, 9000), 2)}, '
            f'"counterparty": "U{rng.randint(10**6, 10**7)}", "flag": {str(rng.random() < 0.2).lower()}}}'
        )
    return f'{{"tool": "{tool}", "rows": [' + ",".join(rows) + "]}"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt corpus: every prompt the LangGraph engine would send for one case
# ─────────────────────────────────────────────────────────────────────────────
def prompts_for_case(
    rng: random.Random,
    idx: int,
    layout: str,
    max_iterations: Optional[int],
    tool_result_bytes: int,
) -> List[Tuple[str, str]]:
    """Return [(node_name, prompt)] in the order the engine would issue them."""
    initial, alert = _case(rng, idx)
    evidence = build_evidence_summary(initial, alert)
    out: List[Tuple[str, str]] = []
    schemas = InvestigationTools.TOOL_SCHEMAS

    def _loop(node: str, system: str, allowed: Sequence[str], iterations: int, max_calls: int):
        messages: List[dict] = []
        lines = tool_lines(schemas, allowed)
        for iteration in range(1, iterations + 1):
            out.append((node, render_react_prompt(
                system=system,
                evidence=evidence,
                tool_lines=lines,
                history=history_block(messages),
                status_lines=status_lines(
                    iteration=iteration,
                    max_iterations=iterations,
                    tool_calls_used=iteration - 1,
                    max_tool_calls=max_calls,
                    hint=default_hint(iteration, iterations),
                ),
                layout=layout,
            )))
            # The loop appends the assistant's tool call and the tool's result,
            # then re-renders — append-only growth, which is the point.
            tool = allowed[(iteration - 1) % len(allowed)]
            messages.append({"role": "assistant", "content": f'{{"tool": "{tool}", "params": {{}}}}'})
            messages.append({
                "role": "tool",
                "tool_name": tool,
                "content": _tool_result_text(rng, tool, tool_result_bytes),
            })

    for name in SPECIALIST_NAMES:
        spec = _SPECIALIST_SPECS[name]
        iterations = min(spec["max_calls"] + 1, max_iterations or 10**6)
        _loop(
            node=name,
            system=_SPECIALIST_SYSTEM.format(
                role=spec["role"], focus=spec["focus"], max_calls=spec["max_calls"], evidence=evidence
            ),
            allowed=_SPECIALIST_TOOLS[name],
            iterations=iterations,
            max_calls=spec["max_calls"],
        )

    findings = "\n\n".join(
        f"### {_SPECIALIST_SPECS[n]['role']}\n- finding {i} for case {idx}: see evidence above."
        for i, n in enumerate(SPECIALIST_NAMES)
    )
    _loop(
        node="llm_agent",
        system=INVESTIGATOR_SYSTEM.format(findings=findings),
        allowed=INVESTIGATOR_TOOLS,
        iterations=min(4, max_iterations or 4),
        max_calls=8,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Block accounting
# ─────────────────────────────────────────────────────────────────────────────
class Tokenizer:
    """Char-estimate tokenizer, or a real HF tokenizer when one is requested."""

    def __init__(self, chars_per_token: float = 4.0, name: Optional[str] = None):
        self.chars_per_token = chars_per_token
        self.impl = None
        if name:
            from transformers import AutoTokenizer  # optional dependency

            self.impl = AutoTokenizer.from_pretrained(name)

    def blocks(self, text: str, block_tokens: int) -> List[str]:
        """Split into block-sized pieces, in token space when a tokenizer is present."""
        if self.impl is not None:
            ids = self.impl(text, add_special_tokens=False)["input_ids"]
            return [
                self.impl.decode(ids[i:i + block_tokens])
                for i in range(0, len(ids), block_tokens)
            ]
        width = max(1, int(block_tokens * self.chars_per_token))
        return [text[i:i + width] for i in range(0, len(text), width)]

    def count(self, text: str) -> int:
        if self.impl is not None:
            return len(self.impl(text, add_special_tokens=False)["input_ids"])
        return int(len(text) / self.chars_per_token)


@dataclass
class Result:
    layout: str
    block_tokens: int
    prompts: int = 0
    blocks_total: int = 0
    blocks_reused: int = 0
    tokens_total: int = 0
    tokens_reused: int = 0
    per_node: Dict[str, List[int]] = field(default_factory=dict)  # node -> [reused, total]

    @property
    def block_reuse_pct(self) -> float:
        return 100.0 * self.blocks_reused / self.blocks_total if self.blocks_total else 0.0

    @property
    def token_reuse_pct(self) -> float:
        return 100.0 * self.tokens_reused / self.tokens_total if self.tokens_total else 0.0

    def as_dict(self) -> dict:
        return {
            "layout": self.layout,
            "block_tokens": self.block_tokens,
            "prompts": self.prompts,
            "blocks_total": self.blocks_total,
            "blocks_reused": self.blocks_reused,
            "block_reuse_pct": round(self.block_reuse_pct, 2),
            "tokens_total": self.tokens_total,
            "tokens_reused": self.tokens_reused,
            "tokens_recomputed": self.tokens_total - self.tokens_reused,
            "token_reuse_pct": round(self.token_reuse_pct, 2),
            "per_node_reuse_pct": {
                node: round(100.0 * r / t, 2) if t else 0.0
                for node, (r, t) in sorted(self.per_node.items())
            },
        }


def measure(
    corpus: Iterable[Tuple[str, str]],
    layout: str,
    block_tokens: int,
    tokenizer: Tokenizer,
) -> Result:
    """Walk the corpus once, counting the leading run of blocks a warm cache holds.

    The cache is a set of chained block hashes shared across every prompt, which is
    how a serving engine's prefix cache (and LMCache behind it) actually behaves.
    Counting stops at the first miss in each prompt: prefix caching cannot resume
    after a mismatch.
    """
    res = Result(layout=layout, block_tokens=block_tokens)
    seen = set()
    for node, prompt in corpus:
        blocks = tokenizer.blocks(prompt, block_tokens)
        chain = hashlib.sha256()
        hit_run = True
        reused_blocks = 0
        reused_chars = 0
        for block in blocks:
            chain.update(block.encode("utf-8"))
            digest = chain.hexdigest()
            if hit_run and digest in seen:
                reused_blocks += 1
                reused_chars += len(block)
            else:
                hit_run = False
            seen.add(digest)

        res.prompts += 1
        res.blocks_total += len(blocks)
        res.blocks_reused += reused_blocks
        res.tokens_total += tokenizer.count(prompt)
        res.tokens_reused += (
            tokenizer.count(prompt[:reused_chars]) if tokenizer.impl is None
            else reused_blocks * block_tokens
        )
        bucket = res.per_node.setdefault(node, [0, 0])
        bucket[0] += reused_blocks
        bucket[1] += len(blocks)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_table(results: List[Result]) -> str:
    head = f"{'layout':<14}{'block':>7}{'prompts':>9}{'blocks':>9}{'reused':>9}{'block %':>10}{'tokens':>11}{'token %':>10}"
    rows = [head, "-" * len(head)]
    for r in results:
        rows.append(
            f"{r.layout:<14}{r.block_tokens:>7}{r.prompts:>9}{r.blocks_total:>9}"
            f"{r.blocks_reused:>9}{r.block_reuse_pct:>9.1f}%{r.tokens_total:>11}{r.token_reuse_pct:>9.1f}%"
        )
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--investigations", type=int, default=20, help="cases in the corpus (default 20)")
    ap.add_argument("--layouts", default=",".join(LAYOUTS),
                    help=f"comma-separated prompt layouts to compare (default: {','.join(LAYOUTS)})")
    ap.add_argument("--block-tokens", default="16",
                    help="comma-separated block sizes in tokens: 16 = vLLM page, 256 = LMCache chunk (default 16)")
    ap.add_argument("--max-iterations", type=int, default=None,
                    help="cap the ReAct iterations per node (default: each node's own max)")
    ap.add_argument("--tool-result-bytes", type=int, default=1200,
                    help="size of each synthetic tool result (default 1200, the history truncation limit)")
    ap.add_argument("--chars-per-token", type=float, default=4.0, help="token estimate (default 4.0)")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer id for exact token blocks (needs `transformers` installed)")
    ap.add_argument("--seed", type=int, default=1337, help="corpus seed (default 1337)")
    ap.add_argument("--json", dest="json_path", default=None, help="also write results as JSON")
    args = ap.parse_args(argv)

    layouts = iter_layouts(args.layouts.split(","))
    block_sizes = [int(b) for b in args.block_tokens.split(",")]
    tokenizer = Tokenizer(args.chars_per_token, args.tokenizer)

    results: List[Result] = []
    for layout in layouts:
        # Same seed per layout: identical cases, only the block order differs.
        rng = random.Random(args.seed)
        corpus = [
            p
            for idx in range(args.investigations)
            for p in prompts_for_case(rng, idx, layout, args.max_iterations, args.tool_result_bytes)
        ]
        for block_tokens in block_sizes:
            results.append(measure(corpus, layout, block_tokens, tokenizer))

    print(f"corpus: {args.investigations} investigations x "
          f"{len(results[0].per_node)} nodes, {results[0].prompts} prompts per layout")
    print(f"tokens: {'tokenizer=' + args.tokenizer if args.tokenizer else f'estimated at {args.chars_per_token} chars/token'}")
    print()
    print(_fmt_table(results))

    by_key = {(r.layout, r.block_tokens): r for r in results}
    base = layouts[0]
    for layout in layouts[1:]:
        for block_tokens in block_sizes:
            a, b = by_key[(base, block_tokens)], by_key[(layout, block_tokens)]
            print(f"\n{layout} vs {base} @ {block_tokens}-token blocks:")
            print(f"  block reuse   {a.block_reuse_pct:5.1f}%  ->{b.block_reuse_pct:6.1f}%   "
                  f"({b.block_reuse_pct - a.block_reuse_pct:+.1f} pts)")
            print(f"  tokens sent   {a.tokens_total:>9,}  ->{b.tokens_total:>10,}   "
                  f"({100.0 * (b.tokens_total - a.tokens_total) / a.tokens_total:+.1f}%)")
            print(f"  recomputed    {a.tokens_total - a.tokens_reused:>9,}  ->"
                  f"{b.tokens_total - b.tokens_reused:>10,}")

    print("\nper-node block reuse %")
    for r in results:
        detail = "  ".join(f"{n}={pct:.1f}%" for n, pct in r.as_dict()["per_node_reuse_pct"].items())
        print(f"  {r.layout:<14} @{r.block_tokens:>4}: {detail}")

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump({"args": vars(args), "results": [r.as_dict() for r in results]}, fh, indent=2)
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
