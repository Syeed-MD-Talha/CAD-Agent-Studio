"""CAD agent harness for CQ-editor.

Pattern taken directly from learn-claude-code s01_agent_loop + s02_tool_use:
    while True:
        response = LLM(messages, tools)
        if no tool_use: break
        execute tools locally (GUI thread - safe for Qt)
        append tool_result

Difference vs s01/s02: tools are CAD-specific, not bash/read/write.
LLM network I/O runs in a QThread (see widgets/chat.py); tool dispatch
runs in the GUI thread so Editor/Debugger access is thread-safe.

.env (repo root or cwd):
    ANTHROPIC_API_KEY=sk-ant-...
    ANTHROPIC_BASE_URL=  (optional, for OpenAI-compatible gateways:
                          MiniMax / GLM / Kimi / DeepSeek / Ollama)
    MODEL_ID=claude-sonnet-4-6
"""

import json
import os
import re
import sys
import traceback
from pathlib import Path
from .skill_store import SKILL_TOOLS, skill_prompt

SYSTEM_PROMPT = """You are a build123d CAD assistant inside CQ-editor.
Generate working build123d Python code (Algebra mode) from the user's natural-language request.

Rules:
1. Always start with `from build123d import *`.
2. Build geometry with Algebra-mode primitives (`Box`, `Cylinder`, `Cone`,
   `Sphere`, `Plane`, `Rectangle`, `Circle`, `extrude`, ...) and operations
   (`fillet`, `chamfer`, faces/edges selectors). End with a variable named `result`.
3. ALWAYS finish with `show_object(result)` so the viewer displays it.
   For multiple bodies use `show_object(part1, name="part1")` etc.
4. Only use build123d API that exists: `Box/Cylinder/extrude`, `fillet`/`chamfer`
   on `part.edges().filter_by(Axis.Z)` style selectors, `Hole`/`CounterBoreHole`
   /`CounterSinkHole`, boolean ops (`part + other`, `part - other`, `part & other`).
5. Keep dimensions parametric and in mm. Reply with a short explanation plus
   ONE ```python fenced block with the full script. Write only the final
   implementation: no abandoned approaches, speculative APIs, or reasoning
   embedded in executable code. Preserve every requested feature.
6. If a tool result contains a traceback, fix the code and try again.
7. Prefer simple, robust code over exotic features.
8. For changing cross-sections, hollow lofts and curved paths, load the
   lofts-and-sweeps skill when skill_view is available.
9. Threads: when the user requests real threads, build real helical geometry;
   do not silently substitute a plain cylinder. Load the threaded-fasteners
   skill when skill_view is available; it links the tested Helix/sweep example.
   If a requested feature cannot be built, explain that limitation honestly.
10. Multi-part requests (bolt + nut, assemblies): build each part separately
    and call `show_object` once per part with a distinct name. For assemblies,
    combine with `Compound(children=[part1, part2])`.
11. Tool workflow: when CAD tools are available, generate the design promptly,
    call `validate_cad_code` with the complete script, then `apply_and_render`
    after validation returns OK. A failed script is NOT in the editor:
    repair the script from the previous response/tool arguments using the
    failing line and API guidance, rather than fetching an empty editor.
    Use get_examples(name="...") to READ an example, not just list names.
    Avoid repeated inspection or delegation instead of generating the design.
    When tools are unavailable or you are told to return code directly,
    return the complete fenced script; the app validates it automatically.
    Use only advertised tools. If apply_and_render is unavailable, return the
    validated script for the parent/app to render instead of claiming a render.
    Both tools REQUIRE a `code` argument with the COMPLETE non-empty Python
    script. Never send empty arguments, null, whitespace, or a partial script.
12. Reference images: the user may attach a sketch or photo. Study the shapes,
    proportions, holes and features visible in the image and reproduce them in
    mm. If dimensions are not readable, choose sensible values and state your
    assumptions in the reply.
13. Visual feedback: after every successful `apply_and_render`, the tool result
    includes a JPEG screenshot of the current 3D view plus the render status.
    If a screenshot is provided, inspect it before replying: compare proportions, holes,
    placement and overall shape against the request. If it looks wrong, fix the
    code and call `validate_cad_code` + `apply_and_render` again (up to the
    iteration limit). Only describe the design as done when the screenshot
    matches the request.

build123d API reference (use these exact call forms):
- Position shapes with `Pos(x, y, z) * shape` or `plane * sketch`.
  `shape.located(...)` and `.move(...)` require a Location, NOT a Plane.
  To use an absolute plane placement: `shape.located(plane.location)`.
- Extrude sketches using the GLOBAL `extrude(sketch, amount=height)`;
  do not use `sketch.extrude(amount=height)`.
- A hexagon is `RegularPolygon(radius, side_count=6)`, with circumradius
  equal to across_flats / sqrt(3). `Polygon` requires explicit vertices.
- Custom offset profiles use `Polygon(*points, align=None)` so their
  coordinates are preserved. `Plane.XZ * Polygon(...)` creates an XZ face.
- Helix uses `Helix(pitch=p, height=h, radius=r)`; a helical ridge uses
  `sweep(profile, path=helix, is_frenet=True)`. Read the thread example.
- Primitives are centered by default. For a shaft starting at z=0 use
  `Cylinder(r, h, align=(Align.CENTER, Align.CENTER, Align.MIN))`.

Example:
```python
from build123d import *
result = Box(30, 20, 10)
result = fillet(result.edges().filter_by(Axis.Z), 2)
show_object(result)
```
"""

# Popular model ids for the picker (editable - any string works).
# Fireworks serverless paths first (today's provider), then Claude direct.
DEFAULT_MODELS = [
    "accounts/fireworks/models/kimi-k2-instruct-0905",
    "accounts/fireworks/models/deepseek-v3p1",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/qwen3p7-plus",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5",
    "gpt-5",
    "gpt-4o",
]

# Fireworks Anthropic-compatible endpoint (NOTE: no trailing /v1, unlike the
# OpenAI-compatible https://api.fireworks.ai/inference/v1).
FIREWORKS_ANTHROPIC_BASE_URL = "https://api.fireworks.ai/inference"


# Reasoning-effort levels (OpenCode/Codex style: low/medium/high/max).
#
# Each level sets TWO things sent to the provider on every LLM call:
#   budget     -> Anthropic `thinking.budget_tokens` target for reasoning
#                 models (None = thinking disabled, fastest/cheapest).
#   max_tokens -> hard ceiling for the whole turn (thinking + answer).
#                 Must stay ABOVE the budget (API rejects budget >= max).
#
# Which models think is provider-defined: currently the Fireworks
# Anthropic-compatible endpoint accepts `thinking` for glm models, so the
# budget only applies there (see thinking_params_for). max_tokens applies
# to every model and raises the output ceiling on high/max.
EFFORTS = {
    "low": {"budget": None, "max_tokens": 8000},
    "medium": {"budget": 3000, "max_tokens": 8000},
    "high": {"budget": 8000, "max_tokens": 16000},
    "max": {"budget": 16000, "max_tokens": 32000},
}
DEFAULT_EFFORT = "medium"


def normalize_effort(effort):
    """Return a valid effort name, falling back to DEFAULT_EFFORT."""
    try:
        name = str(effort or "").strip().lower()
    except Exception:
        return DEFAULT_EFFORT
    return name if name in EFFORTS else DEFAULT_EFFORT


def max_tokens_for_effort(effort=DEFAULT_EFFORT):
    """Output ceiling for an effort level (always above its budget)."""
    return EFFORTS[normalize_effort(effort)]["max_tokens"]


def thinking_params_for(model: str, effort=DEFAULT_EFFORT):
    """Extra `thinking` budget for known over-thinkers, else None.

    Measured on Fireworks (Anthropic endpoint): glm-5p3 spends ~12.6k chars
    of thinking on a single vase prompt (4754 output tokens at max 8000 -
    one follow-up turn pushes it over the edge and the reply is cut with
    no code). A budget of 3000 keeps the same prompt at ~430 output tokens
    with code that still validates. Other models are left untouched.

    Effort mapping (OpenCode/Codex style):
      low    -> thinking off (fastest/cheapest)
      medium -> 3000 (previous default)
      high   -> 8000  (max_tokens raised to 16000)
      max    -> 16000 (max_tokens raised to 32000)
    """
    budget = EFFORTS[normalize_effort(effort)]["budget"]
    if budget is None:
        return None
    if "glm" in (model or "").lower():
        return {"type": "enabled", "budget_tokens": budget}
    return None


# Anthropic tool_use specs (also convertible to OpenAI functions).
TOOLS = [
    {
        "name": "get_editor_code",
        "description": "Get the current text in the CQ-editor code editor.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "validate_cad_code",
        "description": "Syntax-check + dry-run build123d code without touching the viewer. Returns OK or traceback.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "minLength": 1}},
            "required": ["code"],
        },
    },
    {
        "name": "apply_and_render",
        "description": (
            "Write code into the editor and trigger a 3D render. Returns OK or "
            "traceback from the renderer, PLUS an automatic JPEG screenshot of "
            "the 3D view on success so you can visually verify the design and "
            "iterate if it looks wrong."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "minLength": 1}},
            "required": ["code"],
        },
    },
    {
        "name": "get_examples",
        "description": "Read the complete source of a bundled build123d example by name. Omit name to list available examples.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Example stem, e.g. 06_sweep or 09_threaded_bolt_and_nut",
                }
            },
            "required": [],
        },
    },
    {
        "name": "export_model",
        "description": "Export currently rendered build123d models to STL/STEP/BREP file. Returns the saved path or an error.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["stl", "step", "brep"],
                },
                "filename": {
                    "type": "string",
                    "description": "Optional path. Defaults to exports/model.<ext> in cwd.",
                },
            },
            "required": ["format"],
        },
    },
    {
        "name": "spawn_subagent",
        "description": (
            "Delegate a focused subtask to a fresh subagent conversation. "
            "Use this for code generation/repair, validation reasoning, export "
            "planning, or explanation tasks. The subagent sees only the task "
            "and recent parent context; its final text is returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Clear, self-contained instruction for the subagent.",
                },
                "purpose": {
                    "type": "string",
                    "enum": [
                        "code",
                        "validate",
                        "visual_validate",
                        "export",
                        "explain",
                    ],
                    "description": (
                        "What kind of subtask this is. "
                        "visual_validate captures the current 3D viewer screenshot "
                        "and asks the subagent to check the rendered design against "
                        "the request."
                    ),
                },
            },
            "required": ["task", "purpose"],
        },
    },
]
TOOLS.extend(SKILL_TOOLS)

TOOLS.extend(
    [
        {
            "name": "set_design_requirements",
            "description": "Record measurable requirements from the user's request BEFORE modelling. Preserve requested values, units converted to mm, explicit tolerances, and quote each source phrase. These checks are measured independently on completed geometry and shown to the user. Do not invent specifications or relax a failed requirement. Target/other are exact show_object names. ALWAYS set target for a part-specific check (e.g. jar bore target='jar'); use target='all' for the entire assembly. For older blank targets the verifier scopes to an exact unambiguous object name in the requirement text, otherwise all objects. Hole checks cover inward cylindrical bores only (counterbore stages count separately). Unsupported wall-thickness checks remain Not checked.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "requirements": {
                        "type": "array",
                        "maxItems": 100,
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": [
                                        "extent",
                                        "hole_diameter",
                                        "hole_count",
                                        "solid_count",
                                        "object_count",
                                        "min_clearance",
                                        "max_overlap_volume",
                                        "wall_thickness",
                                    ],
                                },
                                "label": {"type": "string"},
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "other": {"type": "string"},
                                "axis": {"type": "string", "enum": ["X", "Y", "Z"]},
                                "expected": {"type": "number", "minimum": 0},
                                "tolerance": {"type": "number", "minimum": 0},
                            },
                            "required": ["kind", "expected", "label", "source"],
                        },
                    }
                },
                "required": ["requirements"],
            },
        },
        {
            "name": "verify_design",
            "description": "Measure the current editor design against recorded requirements in a cancellable isolated process. Returns Passed/Failed/Not checked with dimensions, cylindrical bore measurements, solid counts, overlap and clearance. Does not change the viewport.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "record_visual_review",
            "description": "After inspecting the attached labelled design views against the request/reference, record your AI visual assessment. This does not certify measurements or engineering suitability. Report occluded/uncertain features as Not checked. Available only after view capture for the current design.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["Passed", "Failed", "Not checked"],
                    },
                    "notes": {
                        "type": "string",
                        "description": "Views inspected, matched features, discrepancies, and any remaining uncertainty.",
                    },
                },
                "required": ["status", "notes"],
            },
        },
    ]
)

SYSTEM_PROMPT += """\nDesign verification: Before generating or modifying a model, call set_design_requirements with the measurable requirements from the user, including the exact source phrase. Preserve prior requirements that still apply when modifying a model. Never substitute assumed dimensions for explicit requested values. If only a picture is provided, mark inferred values as assumptions in the source. The user can review and edit the checks. Geometry is measured independently after the build; fix Failed checks without weakening their expected values. Do not claim Not checked items passed. A successful build is not complete design verification. Vision feedback includes labelled isometric, front, top and side views when available; compare each view to the request and call record_visual_review with your assessment and remaining uncertainty. AI visual assessments are separate from measured geometry checks. Exports are reopened and checked, with mesh limitations reported explicitly.\n"""


def build_system_prompt(skills=None, learning=True):
    """Expose skill descriptions here; bodies and examples stay in tool results."""
    return (
        SYSTEM_PROMPT
        + (
            "\nVerification scope: use an explicit show_object name for requirements about one part, "
            "and target='all' only for assembly-wide checks. A displayed object count does not "
            "prove each part is a single solid. If the report flags multi-solid objects, inspect "
            "whether those pieces are intentional; repair disconnected fragments when the user "
            "expects a continuous part and check solid_count on that named part. A verifier error "
            "means the check is unavailable, not that the modelling code failed. Never claim an "
            "unavailable measurement or multi-solid part has been fully verified.\n"
        )
        + skill_prompt(skills, learning=learning)
    )


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """Return the last ```python fenced block, or '' if none."""
    if not text:
        return ""
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return ""


def build_client(model_override="", provider=None, config=None):
    """Build a selected provider profile or the configured .env profile.

    The returned client exposes the Anthropic Messages-shaped interface used by
    the existing CAD agent. OpenAI-compatible providers are adapted internally.
    """
    from .providers import configured_client

    return configured_client(
        provider=provider, model_override=model_override, config=config
    )


def call_llm(client, model, messages, max_tokens=None, effort=DEFAULT_EFFORT):
    """Single LLM turn with tools. Returns the raw Anthropic response."""
    kwargs = {}
    thinking = thinking_params_for(model, effort)
    if thinking is not None:
        kwargs["thinking"] = thinking
    if max_tokens is None:
        max_tokens = max_tokens_for_effort(effort)
    from .recovery import complete

    return complete(
        client,
        model=model,
        system=build_system_prompt(),
        messages=messages,
        tools=TOOLS,
        max_tokens=max_tokens,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# History compaction (pattern from learn-claude-code s08_context_compact).
#
# s01/s02 keep one ever-growing `history` list across turns; s08 adds a
# `prepare()` step before EVERY model call: old tool results are shortened,
# the old middle is archived, and only the recent exchange goes in full.
#
# Why we need it: thinking models (e.g. qwen3p8-max) emit ~15-20k chars of
# <thinking> per turn, and every CAD tool_use input re-sends the FULL script.
# Without compaction turn 2+ resends all of that as input, the model still
# needs a long think, and the reply gets cut at max_tokens (user sees
# "Reply was cut off" and no code). Compaction keeps ids/order intact so
# tool_use <-> tool_result pairing never breaks; only *contents* shrink.
# ---------------------------------------------------------------------------

# Chars to keep for stale (pre-current-turn) content.
_KEEP_TEXT = 1200
_KEEP_CODE = 400
_KEEP_RESULT = 800
_KEEP_USER_TEXT = 1500


def _block_type(block) -> str:
    if isinstance(block, dict):
        return block.get("type", "")
    return getattr(block, "type", "") or ""


def _compact_block(block):
    """Return an API-ready (dict) shortened copy of a stale block,
    or None if the block should be dropped (stale thinking)."""
    btype = _block_type(block)
    if btype in ("thinking", "redacted_thinking"):
        return None  # stale reasoning; current code lives in the editor
    if isinstance(block, dict):
        block = dict(block)  # shallow copy, never mutate stored history
    else:
        # Anthropic SDK object (TextBlock/ToolUseBlock/...) -> plain dict.
        try:
            block = block.model_dump()
        except Exception:
            return block
    if btype == "text":
        text = block.get("text", "") or ""
        if len(text) > _KEEP_TEXT:
            block["text"] = text[:_KEEP_TEXT] + "\n…[older reply truncated]"
    elif btype == "image":
        # Stale reference photos: keep the turn readable, drop the payload.
        return {
            "type": "text",
            "text": "[earlier reference image omitted to save context]",
        }
    elif btype == "tool_use":
        inp = dict(block.get("input") or {})
        code = inp.get("code")
        if isinstance(code, str) and len(code) > _KEEP_CODE:
            inp["code"] = (
                code[:_KEEP_CODE]
                + "\n…[older code truncated — current code is in the editor, "
                "call get_editor_code]"
            )
        else:
            for key, value in list(inp.items()):
                if isinstance(value, str) and len(value) > 500:
                    inp[key] = value[:500] + "…[truncated]"
        block["input"] = inp
    elif btype == "tool_result":
        content = block.get("content", "")
        if isinstance(content, str) and len(content) > _KEEP_RESULT:
            block["content"] = content[:_KEEP_RESULT] + "\n…[older result truncated]"
        elif isinstance(content, list):
            # Stale apply_and_render screenshots: drop heavy image payloads,
            # keep the text verdict so history stays readable.
            compacted_content = []
            for sub in content:
                if isinstance(sub, dict) and sub.get("type") == "image":
                    compacted_content.append(
                        {
                            "type": "text",
                            "text": "[earlier 3D screenshot omitted to save context]",
                        }
                    )
                elif isinstance(sub, dict) and sub.get("type") == "text":
                    text = sub.get("text", "") or ""
                    if len(text) > _KEEP_RESULT:
                        sub = dict(sub)
                        sub["text"] = (
                            text[:_KEEP_RESULT] + "\n…[older result truncated]"
                        )
                    compacted_content.append(sub)
                else:
                    compacted_content.append(sub)
            block["content"] = compacted_content
    return block


def compact_messages(messages):
    """Return an API-ready copy of `messages` with stale turns shortened.

    Only the latest assistant turn (plus everything after it) is passed
    through verbatim, so the in-progress tool_use/tool_result exchange
    always lines up. All older turns are shortened: thinking blocks are
    dropped, old code payloads and old tool results are truncated.
    Ids and ordering are never touched, so pairing never breaks.
    """
    latest_assistant = -1
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message.get("role") == "assistant" and isinstance(
            message.get("content"), list
        ):
            latest_assistant = i
            break
    # No assistant turn yet (fresh prompt): keep the last message verbatim.
    keep_from = latest_assistant if latest_assistant >= 0 else len(messages) - 1
    payload = []
    for index, message in enumerate(messages):
        if index >= keep_from:
            payload.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            if len(content) > _KEEP_USER_TEXT:
                message = dict(message)
                message["content"] = (
                    content[:_KEEP_USER_TEXT] + "\n…[older message truncated]"
                )
            payload.append(message)
        elif isinstance(content, list):
            blocks = []
            for block in content:
                compacted = _compact_block(block)
                if compacted is not None:
                    blocks.append(compacted)
            message = dict(message)
            message["content"] = blocks
            payload.append(message)
        else:
            payload.append(message)
    return payload


def read_example(name="", examples_dir=None):
    """Read only a named bundled example, never an arbitrary model-supplied path."""
    directory = (
        Path(examples_dir)
        if examples_dir
        else Path(__file__).resolve().parents[2] / "examples"
    )
    examples = {path.stem: path for path in sorted(directory.glob("*.py"))}
    if not name:
        return (
            "Examples (call get_examples with a name to read the source): "
            + ", ".join(examples)
        )
    if not isinstance(name, str):
        return "Error: example name must be a string."
    stem = name[:-3] if name.endswith(".py") else name
    if stem not in examples:
        return "Error: unknown bundled example. Available: " + ", ".join(examples)
    return (
        f"Example {stem}:\n```python\n{examples[stem].read_text(encoding='utf-8')}\n```"
    )


def validation_failure(code, detail, line=None, error="", version=""):
    """Put actionable source context ahead of the often long native traceback."""
    lines = code.splitlines()
    source = lines[line - 1].strip() if line and 0 < line <= len(lines) else ""
    summary = (
        error.splitlines()[0][:300] if error else detail.strip().splitlines()[-1][:300]
    )
    heading = f"Traceback: {summary}"
    if line:
        heading += f" (script line {line})"
    if source:
        heading += "\nFailing code: " + source
    hints = []
    if (".located(" in source or ".move(" in source) and "Location" in detail:
        hints.append(
            "located()/move() take a Location, not a Plane. Use shape.located(plane.location) for absolute placement, or plane * sketch / Pos(x, y, z) * shape for transformations."
        )
    if ".extrude(" in source and ("amount" in detail or "argument" in detail):
        hints.append(
            "Use the global extrude(sketch, amount=height), not sketch.extrude(amount=height)."
        )
    if "Polygon(" in source and "RegularPolygon(" not in source:
        hints.append(
            "Polygon requires explicit vertices. A hexagon is RegularPolygon(radius, side_count=6); custom offset vertices need align=None."
        )
    if hints:
        heading += "\nAPI guidance: " + " ".join(hints)
    if version:
        heading += f"\nbuild123d version: {version}"
    return heading + "\n" + detail


def headless_validate(code: str) -> str:
    """Compile + exec code with stubbed show_object/debug/log.

    Runs in-process (needs build123d installed) but does NOT touch Qt.
    Returns 'OK: ...' or 'Traceback: ...' for the agent loop.

    WARNING: executes arbitrary model-generated geometry code in this
    process. A segfault in the OCCT kernel (complex lofts/fillets) kills
    the whole app. Prefer isolated_validate() for agent-driven code.
    """
    try:
        compiled = compile(code, "<cad-agent-validate>", "exec")
    except SyntaxError:
        return "SyntaxError:\n" + traceback.format_exc(limit=3)

    import build123d as _b3d  # noqa: F401

    shown = {}
    namespace = {
        "b3d": _b3d,
        "build123d": _b3d,
        "show_object": lambda obj, name=None, options=None: shown.update(
            {name or f"obj{len(shown)}": obj}
        ),
        "debug": lambda obj, name=None: None,
        "log": lambda x: None,
    }
    try:
        exec(compiled, namespace, namespace)
    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        line = next(
            (
                frame.lineno
                for frame in reversed(frames)
                if frame.filename == "<cad-agent-validate>"
            ),
            None,
        )
        return validation_failure(
            code,
            traceback.format_exc(limit=5),
            line,
            f"{type(error).__name__}: {error}",
            _b3d.__version__,
        )
    if not shown:
        return "WARNING: code ran but show_object() was never called. Add show_object(result)."
    n = len(shown)
    return f"OK: code executed, {n} object(s) shown via show_object()."


# ---------------------------------------------------------------------------
# Isolated pre-render probe: execute model code in a CHILD process so a
# native OCCT segfault (complex geometry) becomes a catchable error string
# instead of killing the whole app. The child mirrors headless_validate
# (same stubs, script dir on sys.path + cwd) and prints one RESULT line.
# ---------------------------------------------------------------------------

#: Default wall-clock budget for the probe child (s). Complex models that
#: legitimately need longer should raise effort/timeout, not the main app.
ISOLATED_PROBE_TIMEOUT = 180

_PROBE_RUNNER = r"""
import json, sys, traceback

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
try:
    compiled = compile(src, "<cad-agent-probe>", "exec")
except SyntaxError:
    print("RESULT " + json.dumps({"ok": False, "kind": "syntax",
          "detail": traceback.format_exc(limit=3)[-4000:]}), flush=True)
    raise SystemExit(0)

import build123d  # noqa: F401

shown = {}
namespace = {
    "b3d": __import__("build123d"),
    "build123d": __import__("build123d"),
    "show_object": lambda obj, name=None, options=None: shown.update(
        {name or f"obj{len(shown)}": obj}
    ),
    "debug": lambda *a, **k: None,
    "log": lambda *a, **k: None,
    "rand_color": lambda *a, **k: {},
}
try:
    exec(compiled, namespace, namespace)
except Exception as error:
    frames = traceback.extract_tb(error.__traceback__)
    line = next((frame.lineno for frame in reversed(frames)
                 if frame.filename == "<cad-agent-probe>"), None)
    print("RESULT " + json.dumps({"ok": False, "kind": "traceback",
          "line": line, "error": f"{type(error).__name__}: {error}",
          "version": build123d.__version__,
          "detail": traceback.format_exc(limit=5)[-4000:]}), flush=True)
    raise SystemExit(0)
for name, obj in shown.items():
    if isinstance(obj, build123d.BuildPart):
        obj = obj.part
    elif isinstance(obj, build123d.BuildSketch):
        obj = obj.sketch
    elif isinstance(obj, build123d.BuildLine):
        obj = obj.line
    try:
        wrapped = getattr(obj, "_wrapped", None)
        valid = (isinstance(obj, build123d.Shape) and wrapped is not None
                 and not wrapped.IsNull() and obj.is_valid and bool(obj.vertices()))
    except Exception:
        valid = False
    if not valid:
        print("RESULT " + json.dumps({"ok": False, "kind": "geometry",
              "detail": f"show_object({name}) received empty, invalid, or non-build123d geometry. Build a valid shape before showing it."}), flush=True)
        raise SystemExit(0)
if not shown:
    print("RESULT " + json.dumps({"ok": True, "warning": True, "objects": 0}),
          flush=True)
else:
    print("RESULT " + json.dumps({"ok": True, "objects": len(shown)}),
          flush=True)
"""


def isolated_validate(code: str, timeout=ISOLATED_PROBE_TIMEOUT, workdir=None) -> str:
    """Validate build123d code in a CHILD process (crash-safe).

    Returns an agent-readable string:
      'OK: ...'        code built N object(s) without crashing
      'WARNING: ...'   ran fine but show_object() was never called
      'SyntaxError:..' / 'Traceback:..'  Python-level failure (fixable)
      'CRASH: ...'     child died (segfault/OOM-kill) -> DO NOT render
                        in-process; ask for a simpler model instead
      'TIMEOUT: ...'   child exceeded `timeout` seconds -> do not render
      'Error: ...'     probe infrastructure failure (no python/build123d)

    `workdir` (script dir) is used as the child cwd and put on sys.path,
    mirroring Debugger._exec, so relative imports keep working.
    """
    import subprocess
    import tempfile

    if not code or not code.strip():
        return "Error: empty code"
    try:
        timeout = int(timeout)
    except Exception:
        timeout = ISOLATED_PROBE_TIMEOUT
    timeout = max(10, timeout)
    if workdir is not None:
        workdir = str(workdir)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_RUNNER, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return (
            f"TIMEOUT: geometry build exceeded {timeout}s in an isolated "
            f"process and was killed. Do NOT render this in-process - "
            f"simplify the model (fewer features, coarser steps) and retry."
        )
    except Exception as e:
        return f"Error: isolated probe could not run: {type(e).__name__}: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    result = None
    try:
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT ") :])
    except Exception:
        result = None
    if result is not None and proc.returncode == 0:
        if result.get("ok"):
            if result.get("warning"):
                return (
                    "WARNING: code ran but show_object() was never "
                    "called. Add show_object(result)."
                )
            return (
                f"OK: code executed in an isolated process, "
                f"{int(result.get('objects', 0))} object(s) shown via "
                f"show_object()."
            )
        detail = result.get("detail", "")
        if result.get("kind") == "syntax":
            return "SyntaxError:\n" + detail
        if result.get("kind") == "geometry":
            return "Error: " + detail
        return validation_failure(
            code,
            detail,
            result.get("line"),
            result.get("error", ""),
            result.get("version", ""),
        )
    # No RESULT line or nonzero exit: the child died natively
    # (segfault in OCCT, OOM-kill, DLL/stack fault). The app survived
    # precisely because this ran out-of-process.
    err_tail = ((proc.stderr or "").strip().splitlines() or [""])[-1][:300]
    return (
        f"CRASH: the geometry kernel died in an isolated process "
        f"(exit {proc.returncode})"
        + (f" - {err_tail}" if err_tail else "")
        + ". Do NOT render this in-process - it would kill the app. "
        f"Simplify the model (fewer/looser features, avoid tiny fillets, "
        f"split into smaller parts) and retry."
    )
