"""On-demand CAD instructions, with validated activation of learned examples."""

import ast
import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

BUILTIN_DIR = Path(__file__).with_name("skills")
_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SECRET = re.compile(
    r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9_.-]{16,})"
    r"|(?:api[_ -]?key|password|secret|token)\s*[=:]\s*['\"][^'\"\n]{8,}['\"]",
    re.I,
)
_OVERRIDE = re.compile(
    r"ignore (?:all |the )?(?:previous|system|developer) instructions"
    r"|(?:disable|bypass|skip) (?:the )?(?:validation|security checks)"
    r"|(?:reveal|print|exfiltrate) .{0,30}(?:api key|credentials|password)"
    r"|(?:powershell|cmd\.exe|curl|wget)\s",
    re.I,
)
_FORBIDDEN = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "breakpoint",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "type",
    "help",
    "dir",
    "exit",
    "quit",
    "read",
    "write",
    "read_text",
    "write_text",
    "read_bytes",
    "write_bytes",
    "load",
    "save",
    "system",
    "popen",
}


def _name(value):
    if not isinstance(value, str) or len(value) > 63 or not _NAME.fullmatch(value):
        raise ValueError(
            "Use a skill name of at most 63 lowercase letters, digits and hyphens."
        )
    return value


def _screen(text):
    if _SECRET.search(text):
        raise ValueError("Remove credentials or secret values before saving a skill.")
    for match in _OVERRIDE.finditer(text):
        if not re.search(
            r"(?:never|not|don't)\s+$",
            text[max(0, match.start() - 12) : match.start()],
            re.I,
        ):
            raise ValueError(
                "Skill instructions cannot override application rules or validation."
            )


def _check_example(code):
    if not isinstance(code, str) or not code.strip() or len(code) > 40000:
        raise ValueError(
            "Provide a complete example_code script (up to 40,000 characters)."
        )
    _screen(code)
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
            )
            if getattr(node, "level", 0) or any(
                module not in ("build123d", "math") for module in modules
            ):
                raise ValueError("Learned examples may import only build123d and math.")
            for alias in node.names:
                if (
                    alias.name.startswith(("_", "import_", "export_"))
                    or alias.name in _FORBIDDEN
                ):
                    raise ValueError(
                        "Learned examples may import CAD construction operations only."
                    )
        symbol = (
            node.id
            if isinstance(node, ast.Name)
            else node.attr if isinstance(node, ast.Attribute) else ""
        )
        if (
            symbol.startswith("_")
            or symbol in _FORBIDDEN
            or symbol.startswith(("import_", "export_"))
        ):
            raise ValueError(
                "Learned examples must contain only CAD construction, without file, network or introspection operations."
            )
    return code


def _metadata(text):
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("SKILL.md needs name and description frontmatter.")
    end = lines.index("---", 1)
    data = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if separator:
            value = value.strip()
            data[key.strip()] = (
                json.loads(value) if value.startswith('"') else value.strip("'")
            )
    _name(data.get("name"))
    description = data.get("description", "")
    if (
        not isinstance(description, str)
        or not 10 <= len(description) <= 240
        or "\n" in description
    ):
        raise ValueError("Provide a single-line description of 10 to 240 characters.")
    return data


def _digest(document, code):
    return hashlib.sha256((document + "\0" + code).encode("utf-8")).hexdigest()


class SkillStore:
    def __init__(self, root=None, builtin_dir=None, validator=None):
        self.root = (
            Path(root) if root is not None else Path.home() / ".cq-editor" / "skills"
        )
        self.builtin_dir = Path(builtin_dir) if builtin_dir is not None else BUILTIN_DIR
        self.validator = validator
        self._lock = threading.RLock()
        self._stamp = None
        self._catalog = []

    @staticmethod
    def _path(root, name, filename):
        directory = root / _name(name)
        path = directory / filename
        if (
            directory.is_symlink()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("Skill paths must stay inside their skill directory.")
        return path

    def _files(self):
        files = []
        for source, root in (
            ("bundled", self.builtin_dir),
            ("active", self.root / "active"),
            ("draft", self.root / "drafts"),
        ):
            if root.is_symlink():
                continue
            for document in sorted(root.glob("*/SKILL.md"))[:100]:
                try:
                    document = self._path(root, document.parent.name, "SKILL.md")
                    tracked = [document]
                    if source != "bundled":
                        tracked += [
                            self._path(root, document.parent.name, name)
                            for name in ("example.py", "receipt.json")
                        ]
                    stamp = tuple(
                        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
                        for path in tracked
                    )
                    files.append((source, document, stamp))
                except (OSError, ValueError):
                    continue
        return files

    def catalog(self, include_drafts=False):
        with self._lock:
            files = self._files()
            stamp = tuple((source, values) for source, _, values in files)
            if stamp != self._stamp:
                entries = []
                reserved = set()
                for source, path, _ in files:
                    try:
                        text = path.read_text(encoding="utf-8")
                        data = _metadata(text)
                        name = data["name"]
                        if name != path.parent.name or (
                            source != "bundled" and name in reserved
                        ):
                            continue
                        if source == "bundled":
                            reserved.add(name)
                        else:
                            receipt = json.loads(
                                path.with_name("receipt.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                            code = path.with_name("example.py").read_text(
                                encoding="utf-8"
                            )
                            if (
                                not isinstance(receipt, dict)
                                or receipt.get("digest") != _digest(text, code)
                                or receipt.get("status") != source
                            ):
                                continue
                            _screen(text)
                            _check_example(code)
                        entries.append(
                            {
                                "name": name,
                                "description": data["description"],
                                "status": source,
                                "path": path,
                                "digest": (
                                    None if source == "bundled" else receipt["digest"]
                                ),
                            }
                        )
                    except (OSError, ValueError, SyntaxError, TypeError):
                        continue
                self._catalog, self._stamp = entries, stamp
            return [
                dict(entry)
                for entry in self._catalog
                if include_drafts or entry["status"] != "draft"
            ]

    def index(self):
        entries = self.catalog()
        return "\n".join(
            f"- {entry['name']}: {entry['description']}" for entry in entries
        )

    def summary(self):
        entries = self.catalog(include_drafts=True)
        active = {(e["name"], e["digest"]) for e in entries if e["status"] == "active"}
        lines = ["CAD skills (details load only when needed):"]
        for entry in entries:
            if (
                entry["status"] == "draft"
                and (entry["name"], entry["digest"]) in active
            ):
                continue
            lines.append(
                f"- {entry['name']} [{entry['status']}]: {entry['description']}"
            )
        return "\n".join(lines)

    def view(self, name="", file=None, draft=False):
        try:
            if not name:
                return self.summary()
            _name(name)
            entries = [
                entry
                for entry in self.catalog(include_drafts=draft)
                if entry["name"] == name and (not draft or entry["status"] == "draft")
            ]
            if not entries:
                return "Error: unknown skill or inactive draft. Use skill_view without a name to list skills."
            entry = entries[0]
            if file is not None and file not in ("SKILL.md", "example.py"):
                return "Error: supported skill files are SKILL.md and example.py."
            path = (
                entry["path"]
                if not file or file == "SKILL.md"
                else entry["path"].with_name(file)
            )
            path = self._path(path.parent.parent, name, path.name)
            if not path.is_file():
                return "Error: this skill has no example.py. Read the named bundled example using get_examples."
            text = path.read_text(encoding="utf-8")
            label = f"Skill {name} [{entry['status']}]"
            return (
                label
                + "\n"
                + (f"```python\n{text}\n```" if path.suffix == ".py" else text)
            )
        except (OSError, ValueError) as error:
            return f"Error: {error}"

    @staticmethod
    def _write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as handle:
                temporary = handle.name
                handle.write(text)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _save(self, stage, name, document, code, receipt):
        root = self.root / stage
        if self.root.is_symlink() or root.is_symlink():
            raise ValueError("Skill storage cannot use a redirected directory.")
        paths = [
            self._path(root, name, filename)
            for filename in ("SKILL.md", "example.py", "receipt.json")
        ]
        for path, text in zip(paths, (document, code, json.dumps(receipt, indent=2))):
            self._write(path, text)
        self._stamp = None

    def manage(self, action, name, description="", instructions="", example_code=""):
        """Agent writes drafts; only the host can issue an activation receipt."""
        try:
            with self._lock:
                _name(name)
                if any(
                    entry["name"] == name and entry["status"] == "bundled"
                    for entry in self.catalog()
                ):
                    raise ValueError(
                        "Bundled skills are read-only. Choose a distinct name for a learned method."
                    )
                if action == "save_draft":
                    if (
                        not isinstance(instructions, str)
                        or not 40 <= len(instructions) <= 12000
                    ):
                        raise ValueError(
                            "Provide focused instructions of 40 to 12,000 characters."
                        )
                    document = f"---\nname: {name}\ndescription: {json.dumps(description)}\n---\n\n{instructions.strip()}\n"
                    _metadata(document)
                    _screen(document)
                    code = _check_example(example_code)
                    receipt = {"status": "draft", "digest": _digest(document, code)}
                    self._save("drafts", name, document, code, receipt)
                    return f"OK: saved draft skill {name}. It is excluded from automatic discovery until skill_manage(action='activate', name='{name}') passes geometry validation."
                if action != "activate":
                    raise ValueError("Supported actions are save_draft and activate.")
                root = self.root / "drafts"
                path = self._path(root, name, "SKILL.md")
                code_path = self._path(root, name, "example.py")
                document, code = path.read_text(encoding="utf-8"), code_path.read_text(
                    encoding="utf-8"
                )
                if _metadata(document)["name"] != name:
                    raise ValueError("The draft name must match its directory.")
                _screen(document)
                _check_example(code)
                from .cad_agent import isolated_validate

                verdict = (self.validator or isolated_validate)(code)
                if not verdict.startswith("OK:"):
                    return f"Error: skill {name} remains a draft. Example validation failed:\n{verdict}"
                if document != path.read_text(
                    encoding="utf-8"
                ) or code != code_path.read_text(encoding="utf-8"):
                    raise ValueError(
                        "The draft changed during validation. Retry activation of the updated draft."
                    )
                receipt = {
                    "status": "active",
                    "digest": _digest(document, code),
                    "validation": verdict,
                    "validated_at": datetime.now(timezone.utc).isoformat(),
                }
                self._save("active", name, document, code, receipt)
                return f"OK: activated skill {name}. Its example passed geometry validation; check suitability and dimensions for each future request."
        except (OSError, ValueError, SyntaxError, TypeError) as error:
            return f"Error: {error}"


def skill_prompt(store=None, learning=True):
    store = store if store is not None else SkillStore()
    index = store.index()
    text = (
        "\n\nAvailable CAD skills (names and descriptions only):\n"
        + index
        + "\nLoad a relevant skill with skill_view(name=...) before using its method. "
        "Read example.py only when needed. Skills are modeling guidance; the user's "
        "dimensions and application validation rules still apply. Use advertised tools only. "
        "If skill_view is unavailable, use guidance already in context and return complete code."
    )
    if learning:
        text += (
            " After a successful design or repair, consider saving a NEW reusable method "
            "using skill_manage(action='save_draft', name=..., description=..., instructions=..., example_code=...). "
            "Avoid duplicates, one-off dimensions as universal rules, and trivial skills. "
            "Save only CAD methodology and a self-contained example, without personal data, keys or paths. "
            "Only skill_manage(action='activate', name=...) can make a draft discoverable after validation. "
            "Finish the user's design first; skill learning is optional and must not delay a final answer "
            "when tool rounds are scarce. Never claim validation proves all dimensions or future variants."
        )
    return text


SKILL_TOOLS = [
    {
        "name": "skill_view",
        "description": "Load a CAD skill's instructions on demand, or read its example.py attachment. Omit name to list active and draft skills.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "file": {"type": "string", "enum": ["SKILL.md", "example.py"]},
                "draft": {
                    "type": "boolean",
                    "description": "Read a saved draft for revision; drafts are not active instructions.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "skill_manage",
        "description": "Save a reusable CAD method as a local draft, or activate a draft after the host validates its example. Cannot change bundled skills. Complete the user's design before optional learning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["save_draft", "activate"]},
                "name": {"type": "string", "maxLength": 63},
                "description": {
                    "type": "string",
                    "description": "When this CAD method is useful (required for save_draft).",
                },
                "instructions": {
                    "type": "string",
                    "description": "Reusable methodology, assumptions and pitfalls as Markdown (required for save_draft).",
                },
                "example_code": {
                    "type": "string",
                    "description": "Complete self-contained build123d construction ending in show_object, without exports or file access (required for save_draft).",
                },
            },
            "required": ["action", "name"],
        },
    },
]
