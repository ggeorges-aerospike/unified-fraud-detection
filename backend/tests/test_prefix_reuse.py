"""
Prompt-layout and prefix-reuse tests (no LLM, no Aerospike, no docker).

Two things are worth locking down:

1. The ``current`` layout must stay byte-identical to the prompt the LangGraph
   engine has always sent. It is the baseline every reuse measurement is compared
   against, and a stray newline would silently invalidate it.
2. ``stable_first`` must actually do the two things it exists for — move the
   per-iteration STATUS block to the tail, and drop the duplicated evidence block.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.prefix_reuse import Tokenizer, measure, prompts_for_case
from workflow.prompt_layout import (
    CURRENT,
    STABLE_FIRST,
    active_layout,
    blocks_for_layout,
    default_hint,
    history_block,
    render_react_prompt,
    status_lines,
    tool_lines,
)


class TestRenderCurrentLayout(unittest.TestCase):
    def test_current_layout_is_byte_identical_to_the_historical_prompt(self):
        system, evidence, history = "SYSTEM TEXT", "EVIDENCE TEXT", "HISTORY TEXT"
        lines = ["- tool_a: does a params {}", "- tool_b: does b params {}"]
        status = status_lines(2, 8, 1, 12, "HINT")

        expected = f"""{system}

## CASE EVIDENCE
{evidence}

## TOOLS (JSON format only)
{chr(10).join(lines)}

## STATUS
- Iteration: 2/8
- Tool calls: 1/12
- HINT

## CONVERSATION
{history}

YOUR JSON RESPONSE:"""

        self.assertEqual(
            render_react_prompt(system, evidence, lines, history, status, layout=CURRENT),
            expected,
        )

    def test_default_layout_is_current(self):
        os.environ.pop("PROMPT_LAYOUT", None)
        self.assertEqual(active_layout(), CURRENT)
        self.assertEqual(blocks_for_layout(CURRENT)[-1], "conversation")

    def test_unknown_layout_falls_back_rather_than_raising(self):
        self.assertEqual(active_layout("nonsense"), CURRENT)


class TestStableFirstLayout(unittest.TestCase):
    def _render(self, system, evidence):
        return render_react_prompt(
            system, evidence, ["- tool_a: does a params {}"], "HISTORY",
            status_lines(2, 8, 1, 12, "HINT"), layout=STABLE_FIRST,
        )

    def test_status_block_is_last(self):
        out = self._render("SYSTEM", "EVIDENCE")
        self.assertLess(out.index("## CONVERSATION"), out.index("## STATUS"))
        self.assertEqual(blocks_for_layout(STABLE_FIRST)[-1], "status")

    def test_duplicate_evidence_block_is_dropped(self):
        # The specialist template already ends with its own CASE EVIDENCE section.
        body = "acct A0001 balance $5,000"
        system = f"SYSTEM\n\n## CASE EVIDENCE\n{body}"
        out = self._render(system, body)
        self.assertEqual(out.count("## CASE EVIDENCE"), 1)
        self.assertEqual(out.count(body), 1)

    def test_evidence_block_kept_when_not_already_in_system(self):
        out = self._render("SYSTEM", "EVIDENCE")
        self.assertIn("## CASE EVIDENCE\nEVIDENCE", out)


class TestPromptBlockHelpers(unittest.TestCase):
    def test_submit_assessment_is_always_offered(self):
        schemas = [
            {"name": "get_account_transactions", "description": "d", "parameters": {"a": 1}},
            {"name": "submit_assessment", "description": "d", "parameters": {}},
            {"name": "detect_fraud_ring", "description": "d", "parameters": {}},
        ]
        rendered = tool_lines(schemas, ["get_account_transactions"])
        self.assertEqual(len(rendered), 2)
        self.assertTrue(any("submit_assessment" in line for line in rendered))
        self.assertFalse(any("detect_fraud_ring" in line for line in rendered))

    def test_hint_switches_at_the_iteration_limit(self):
        self.assertIn("JSON object", default_hint(1, 8))
        self.assertIn("iteration limit", default_hint(7, 8))

    def test_empty_history_is_stable_text(self):
        self.assertEqual(history_block([]), "(First iteration - no history yet)")

    def test_history_is_append_only(self):
        first = [{"role": "assistant", "content": "a"}]
        second = first + [{"role": "tool", "tool_name": "t", "content": "r"}]
        self.assertTrue(history_block(second).startswith(history_block(first)))


class TestReuseMeasurement(unittest.TestCase):
    def test_repeated_identical_prompt_is_almost_entirely_reused(self):
        tok = Tokenizer(chars_per_token=4.0)
        prompt = "X" * 4000
        res = measure([("n", prompt)] * 5, CURRENT, 16, tok)
        # First pass is all misses, the other four are hits.
        self.assertGreater(res.block_reuse_pct, 75.0)

    def test_a_changed_leading_byte_invalidates_the_whole_prompt(self):
        tok = Tokenizer(chars_per_token=4.0)
        base = "A" + "X" * 4000
        changed = "B" + "X" * 4000
        res = measure([("n", base), ("n", changed)], CURRENT, 16, tok)
        self.assertEqual(res.blocks_reused, 0)

    def test_stable_first_beats_current_on_the_real_prompts(self):
        tok = Tokenizer(chars_per_token=4.0)
        scores = {}
        for layout in (CURRENT, STABLE_FIRST):
            rng = random.Random(7)
            corpus = [
                p for idx in range(3)
                for p in prompts_for_case(rng, idx, layout, max_iterations=4, tool_result_bytes=600)
            ]
            scores[layout] = measure(corpus, layout, 16, tok)

        self.assertGreater(
            scores[STABLE_FIRST].block_reuse_pct, scores[CURRENT].block_reuse_pct,
            "stable_first should raise prefix reuse; if this fails the layout regressed",
        )
        # It also sends strictly fewer tokens, because the duplicate evidence is gone.
        self.assertLess(scores[STABLE_FIRST].tokens_total, scores[CURRENT].tokens_total)

    def test_corpus_covers_every_node(self):
        rng = random.Random(1)
        nodes = {node for node, _ in prompts_for_case(rng, 0, CURRENT, 3, 400)}
        self.assertEqual(
            nodes, {"network_analyst", "device_analyst", "velocity_analyst", "llm_agent"}
        )


if __name__ == "__main__":
    unittest.main()
