"""
ReAct prompt block assembly, with a switchable block order.

Prefix caching (vLLM's own, and any KV-cache tier behind it such as LMCache with
Aerospike as the L2/L3 backend) only matches from the **start** of the prompt, so
the order of the blocks decides how much of each prompt is reusable:

- ``current``      — the historical layout: system, evidence, tools, status, conversation.
                     ``## STATUS`` carries the iteration counter and sits directly in
                     front of the append-only conversation, so iteration N+1 shares no
                     prefix with iteration N past that point and the whole history is
                     re-prefilled every iteration.
- ``stable_first`` — most-stable-first: system, tools, evidence, conversation, status.
                     Everything that mutates per iteration moves to the tail, so only
                     the tail re-prefills. Also drops the duplicated evidence block
                     when the system prompt already carries the same text.

Selected by ``PROMPT_LAYOUT`` (default ``current``, i.e. no behaviour change until a
measurement justifies flipping it). ``benchmarks/prefix_reuse.py`` measures both.

This module is intentionally dependency-free so the benchmark can import it without
pulling in the LLM/graph/database stack.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

CURRENT = "current"
STABLE_FIRST = "stable_first"
LAYOUTS = (CURRENT, STABLE_FIRST)

DEFAULT_LAYOUT = CURRENT

TRAILER = "YOUR JSON RESPONSE:"

# Block order per layout. "system" is the leading block and carries no header.
_ORDER = {
    CURRENT: ("system", "evidence", "tools", "status", "conversation"),
    STABLE_FIRST: ("system", "tools", "evidence", "conversation", "status"),
}

_HEADERS = {
    "evidence": "## CASE EVIDENCE",
    "tools": "## TOOLS (JSON format only)",
    "status": "## STATUS",
    "conversation": "## CONVERSATION",
}


def active_layout(layout: Optional[str] = None) -> str:
    """Resolve the layout: explicit argument, then ``PROMPT_LAYOUT``, then default."""
    name = (layout or os.environ.get("PROMPT_LAYOUT") or DEFAULT_LAYOUT).strip().lower()
    return name if name in LAYOUTS else DEFAULT_LAYOUT


def render_react_prompt(
    system: str,
    evidence: str,
    tool_lines: Sequence[str],
    history: str,
    status_lines: Sequence[str],
    layout: Optional[str] = None,
) -> str:
    """Assemble the ReAct prompt from its blocks in the configured order.

    ``current`` reproduces the historical prompt byte for byte.
    """
    name = active_layout(layout)

    bodies = {
        "system": system,
        "evidence": evidence,
        "tools": "\n".join(tool_lines),
        "status": "\n".join(status_lines),
        "conversation": history,
    }

    skip = set()
    if name == STABLE_FIRST and evidence.strip() and evidence.strip() in system:
        # The caller's system prompt already ends with its own CASE EVIDENCE section
        # (the specialist template does); a second copy is pure token waste.
        skip.add("evidence")

    parts: List[str] = []
    for block in _ORDER[name]:
        if block in skip:
            continue
        header = _HEADERS.get(block)
        parts.append(f"{header}\n{bodies[block]}" if header else bodies[block])

    return "\n\n".join(parts) + f"\n\n{TRAILER}"


def history_block(messages: Sequence[dict]) -> str:
    """Render the CONVERSATION block: append-only, and byte-stable once written.

    Truncation limits (500 chars per assistant turn, 1200 per tool result) are
    part of the cache contract — changing them invalidates every cached prefix.
    """
    lines = []
    for msg in messages:
        if msg["role"] == "assistant":
            lines.append(f"\nAssistant: {msg['content'][:500]}")
        elif msg["role"] == "tool":
            lines.append(f"\nTool Result ({msg.get('tool_name', 'unknown')}): {msg['content'][:1200]}")
    return "".join(lines) if lines else "(First iteration - no history yet)"


def tool_lines(schemas: Sequence[dict], allowed_names: Iterable[str]) -> List[str]:
    """Render the TOOLS block, one line per allowed tool, in schema order.

    ``submit_assessment`` is always allowed — it is the loop's exit.
    """
    wanted = set(allowed_names)
    lines = []
    for schema in schemas:
        name = schema["name"]
        if name not in wanted and name != "submit_assessment":
            continue
        params = ", ".join(f'"{k}": ...' for k in schema.get("parameters", {}))
        lines.append(f'- {name}: {schema["description"]} params {{{params}}}')
    return lines


def default_hint(iteration: int, max_iterations: int, submit_hint: str = "") -> str:
    """The response-format hint that trails the STATUS block."""
    if iteration >= max_iterations - 1:
        return "You are at the iteration limit. Call submit_assessment now if available."
    return submit_hint or (
        'Respond with ONLY a JSON object: {"tool": "<name>", "params": {...}}'
    )


def status_lines(
    iteration: int,
    max_iterations: int,
    tool_calls_used: int,
    max_tool_calls: int,
    hint: str,
) -> List[str]:
    """The STATUS block — every line here mutates per iteration."""
    return [
        f"- Iteration: {iteration}/{max_iterations}",
        f"- Tool calls: {tool_calls_used}/{max_tool_calls}",
        f"- {hint}",
    ]


def blocks_for_layout(layout: Optional[str] = None) -> Tuple[str, ...]:
    """The block order in use — for reporting and tests."""
    return _ORDER[active_layout(layout)]


def iter_layouts(names: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
    """Validate and normalise a list of layout names."""
    if not names:
        return LAYOUTS
    out = []
    for n in names:
        n = n.strip().lower()
        if n not in LAYOUTS:
            raise ValueError(f"unknown layout '{n}'; expected one of {', '.join(LAYOUTS)}")
        out.append(n)
    return tuple(out)
