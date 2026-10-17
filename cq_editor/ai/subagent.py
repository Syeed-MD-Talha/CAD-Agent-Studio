"""Subagent harness for CQ-editor's AI chat.

Pattern taken from learn-claude-code s06_subagent + s15_integrated_harness:
    - a subagent is a fresh, short LLM conversation for a focused task
    - only the final text returns to the parent conversation
    - subagents do NOT recurse (no task tool inside a subagent)
    - all network I/O is synchronous; callers run this off the GUI thread

The helper is intentionally simple so it can be called from QThread workers
or from the main thread during development.
"""

from __future__ import annotations

import base64
from typing import Any

from .cad_agent import (
    SYSTEM_PROMPT,
    TOOLS,
    build_client,
    compact_messages,
    max_tokens_for_effort,
    normalize_effort,
    thinking_params_for,
)


SUBAGENT_SYSTEM = (
    "You are a focused build123d coding subagent inside CQ-editor. "
    "Complete the task given by the parent agent, then return a concise "
    "final text answer. Do not ask follow-up questions. "
    "Do not spawn additional subagents."
)

# Subagents get a cheaper/faster default model when the env var is set.
# Falls back to the same model the parent is using.
SUBAGENT_MODEL_ENV = "CQ_SUBAGENT_MODEL_ID"

# Cap subagent turns so a runaway loop cannot hang forever.
MAX_SUBAGENT_TURNS = 8


def _extract_text(content) -> str:
    """Visible text from an Anthropic content list."""
    if not isinstance(content, list):
        return str(content)
    parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        else:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
    return "\n".join(parts).strip()


def _has_tool_use(content) -> bool:
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "tool_use":
                return True
        else:
            if getattr(block, "type", None) == "tool_use":
                return True
    return False


def run_subagent(
    task: str,
    system_prompt: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    parent_messages: list[dict[str, Any]] | None = None,
    compact_parent: bool = True,
    image_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Run a focused subagent and return a structured result.

    Args:
        task: the user-facing instruction for the subagent.
        system_prompt: override the default subagent system prompt.
        model: explicit model id; defaults to SUBAGENT_MODEL_ENV or the
            model selected by build_client().
        effort: reasoning effort (low/medium/high/max); defaults to medium.
        max_tokens: output ceiling; defaults to effort-based ceiling.
        tools: tool schema available to the subagent. Defaults to the full
            parent TOOLS set so the subagent can validate/inspect code.
        parent_messages: optional parent conversation context. When provided,
            the latest user/assistant turns are appended after compaction so
            the subagent understands the design history without seeing every
            stale reasoning block.
        compact_parent: if True, stale parent turns are shortened before
            being shown to the subagent.

    Returns:
        {"ok": bool, "text": str, "usage": {"in": int, "out": int}}
    """
    import os

    client, default_model = build_client(model_override=model or "")
    model = model or os.getenv(SUBAGENT_MODEL_ENV, "") or default_model
    effort = normalize_effort(effort)
    if max_tokens is None:
        max_tokens = max_tokens_for_effort(effort)
    thinking = thinking_params_for(model, effort)

    system = (system_prompt or SUBAGENT_SYSTEM).strip()
    messages: list[dict[str, Any]] = []

    if parent_messages:
        ctx = compact_messages(parent_messages) if compact_parent else list(parent_messages)
        # Keep only the most recent exchange so the subagent has context
        # without being drowned in old tool results.
        keep = []
        for msg in reversed(ctx):
            keep.append(msg)
            if msg.get("role") == "user" and len(keep) >= 4:
                break
        if keep:
            messages.extend(reversed(keep))

    user_content: list[dict[str, Any]] = [{"type": "text", "text": task}]
    if image_bytes:
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(image_bytes).decode("ascii"),
            },
        })
    messages.append({"role": "user", "content": user_content})

    kwargs = {}
    if thinking is not None:
        kwargs["thinking"] = thinking

    tool_schema = tools if tools is not None else TOOLS

    total_in = 0
    total_out = 0

    try:
        for _ in range(MAX_SUBAGENT_TURNS):
            response = client.messages.create(
                model=model,
                system=system,
                messages=messages,
                tools=tool_schema,
                max_tokens=max_tokens,
                **kwargs,
            )
            usage = getattr(response, "usage", None)
            total_in += int(getattr(usage, "input_tokens", 0) or 0)
            total_out += int(getattr(usage, "output_tokens", 0) or 0)

            messages.append({"role": "assistant", "content": response.content})

            if not _has_tool_use(response.content):
                text = _extract_text(response.content)
                return {
                    "ok": True,
                    "text": text or "(subagent returned no text)",
                    "usage": {"in": total_in, "out": total_out},
                }

            # Subagents do not recurse: return a clear refusal for any
            # spawn_subagent tool call.
            results = []
            for block in response.content:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if btype != "tool_use":
                    continue
                name = block.get("name") if isinstance(block, dict) else getattr(block, "name", "")
                if name == "spawn_subagent":
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.get("id") if isinstance(block, dict) else getattr(block, "id", ""),
                        "content": "Error: subagents cannot spawn further subagents.",
                    })
                else:
                    # The subagent asked for a tool we cannot run here.
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.get("id") if isinstance(block, dict) else getattr(block, "id", ""),
                        "content": f"Error: tool '{name}' is not available to subagents.",
                    })
            messages.append({"role": "user", "content": results})

        return {
            "ok": False,
            "text": "Subagent stopped after maximum turns without a final answer.",
            "usage": {"in": total_in, "out": total_out},
        }
    except Exception as e:
        return {
            "ok": False,
            "text": f"Subagent failed: {type(e).__name__}: {e}",
            "usage": {"in": total_in, "out": total_out},
        }


def run_code_subagent(
    request: str,
    current_code: str = "",
    parent_messages: list[dict[str, Any]] | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    """Specialised subagent: generate or fix build123d code.

    Returns the same dict as run_subagent(); the generated code is inside
    the "text" field as a ```python fenced block.
    """
    task_lines = [
        "Generate working build123d Python code (Algebra mode) for the request below.",
        "Rules:",
        "- Start with `from build123d import *`.",
        "- End with a variable named `result` and call `show_object(result)`.",
        "- Use only existing build123d API (Box/Cylinder/extrude, fillet/chamfer, Hole).",
        "- Reply with a short explanation plus ONE ```python fenced block.",
    ]
    if current_code and current_code.strip():
        task_lines.append("\nCurrent editor code (modify or replace it as needed):")
        task_lines.append(f"```python\n{current_code}\n```")
    task_lines.append("\nRequest:")
    task_lines.append(request)

    return run_subagent(
        task="\n".join(task_lines),
        system_prompt=SYSTEM_PROMPT,
        model=model,
        effort=effort,
        tools=None,
        parent_messages=parent_messages,
    )
