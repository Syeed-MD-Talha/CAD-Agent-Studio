"""AI chatbot agent for CQ-editor.

Adapted from learn-claude-code harness pattern:
  agent_loop + tools + permissions, but with CAD-specific tools.
"""

from .cad_agent import (
    SYSTEM_PROMPT,
    build_system_prompt,
    TOOLS,
    DEFAULT_MODELS,
    EFFORTS,
    DEFAULT_EFFORT,
    normalize_effort,
    max_tokens_for_effort,
    extract_python_code,
    build_client,
    headless_validate,
    isolated_validate,
    ISOLATED_PROBE_TIMEOUT,
    compact_messages,
    thinking_params_for,
)
from .sessions import SessionStore, autotitle, serialize_messages
from .skill_store import SkillStore

__all__ = [
    "SYSTEM_PROMPT",
    "build_system_prompt",
    "SkillStore",
    "TOOLS",
    "DEFAULT_MODELS",
    "EFFORTS",
    "DEFAULT_EFFORT",
    "normalize_effort",
    "max_tokens_for_effort",
    "extract_python_code",
    "build_client",
    "headless_validate",
    "isolated_validate",
    "ISOLATED_PROBE_TIMEOUT",
    "compact_messages",
    "thinking_params_for",
    "SessionStore",
    "autotitle",
    "serialize_messages",
]
