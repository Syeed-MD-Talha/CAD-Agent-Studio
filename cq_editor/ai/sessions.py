"""Chat session management for CQ-editor's AI panel.

Pattern taken from learn-claude-code:
  s08_context_compact -> transcripts persisted per session (.transcripts/*.jsonl)
  s09_memory          -> file-backed store with an index (MEMORY.md catalog)

Each chat session is one `<uuid>.json` file under ~/.cq-editor/chat_sessions/
holding the full Anthropic `messages` list (assistant SDK blocks are stored
as plain dicts via model_dump, so they reload API-ready). An `index.json`
keeps id/title/updated/preview for the session switcher without loading
every transcript.
"""

import json
import time
import uuid
from pathlib import Path

SESSIONS_DIR = Path.home() / ".cq-editor" / "chat_sessions"
INDEX_FILE = "index.json"
MAX_SESSIONS = 50

# ChatGPT/Codex-style default. Fresh sessions keep this title until the
# first user message auto-titles them (see autotitle/is_auto_title).
DEFAULT_TITLE = "New chat"
# Titles from older versions that were also auto-generated. Recognised as
# "untitled" so the first real prompt still renames the session.
AUTO_TITLES = {"New chat", "New design", "Chat", ""}


def _utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def serialize_messages(messages):
    """Convert stored messages (SDK blocks or dicts) to plain JSON data."""
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict):
                    blocks.append(block)
                else:
                    try:
                        blocks.append(block.model_dump())
                    except Exception:
                        blocks.append({"type": "text", "text": str(block)})
            message = dict(message)
            message["content"] = blocks
        out.append(message)
    return out


def session_preview(messages, limit=120):
    """One-line preview for the switcher: first user text found."""
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user":
            continue
        if isinstance(content, str) and content.strip():
            text = " ".join(content.split())
            return text[:limit]
        if isinstance(content, list):
            for block in content:
                btype = block.get("type") if isinstance(block, dict) else ""
                text = block.get("text", "") if isinstance(block, dict) else ""
                if btype == "text" and text.strip():
                    return " ".join(text.split())[:limit]
    return ""


def autotitle(messages, limit=42):
    """ChatGPT-style title: first user line, word-boundary truncated.

    Falls back to DEFAULT_TITLE when there is no user text yet (e.g.
    image-only prompts get "Reference images" instead of an empty title).
    """
    preview = session_preview(messages, limit=200)
    if not preview:
        # Image-only or empty prompt: did the user attach reference images?
        try:
            for message in messages or []:
                content = message.get("content")
                if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "image" for b in content
                ):
                    return "Reference images"
        except Exception:
            pass
        return DEFAULT_TITLE
    # First line only (ChatGPT/Codex use the opening words, not the whole dump).
    first_line = preview.splitlines()[0] if "\n" in preview else preview
    first_line = " ".join(first_line.split()).strip(" -–—:;,.!?\"'“”‘’")
    if not first_line:
        return DEFAULT_TITLE
    if len(first_line) <= limit:
        return first_line
    # Word-boundary cut so titles never end mid-word.
    cut = first_line[:limit].rsplit(" ", 1)[0] or first_line[:limit]
    return cut.rstrip(" -–—:;,.!?\"'“”‘’") + "…"


def is_auto_title(title):
    """True when the title is still a placeholder (safe to auto-rename)."""
    try:
        return " ".join(str(title or "").split()) in AUTO_TITLES
    except Exception:
        return True


class SessionStore:
    """File-backed chat sessions with an index (s09-style)."""

    def __init__(self, base_dir=None):
        self.base = Path(base_dir) if base_dir else SESSIONS_DIR
        self.base.mkdir(parents=True, exist_ok=True)

    # ------------- index -------------
    def _index_path(self):
        return self.base / INDEX_FILE

    def _read_index(self):
        try:
            data = json.loads(self._index_path().read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict) and e.get("id")]
        except Exception:
            pass
        return []

    def _write_index(self, entries):
        try:
            self._index_path().write_text(
                json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except Exception:
            pass

    def list_sessions(self):
        """Newest-first [{id, title, updated, preview}]."""
        entries = self._read_index()
        entries.sort(key=lambda e: e.get("updated", ""), reverse=True)
        return entries

    def _upsert_index(self, session_id, title, preview):
        entries = [e for e in self._read_index() if e.get("id") != session_id]
        entries.append(
            {"id": session_id, "title": title, "updated": _utcnow(), "preview": preview}
        )
        entries.sort(key=lambda e: e.get("updated", ""), reverse=True)
        # Prune oldest beyond MAX_SESSIONS (s08-style snip).
        for stale in entries[MAX_SESSIONS:]:
            try:
                (self.base / f"{stale['id']}.json").unlink()
            except Exception:
                pass
        self._write_index(entries[:MAX_SESSIONS])

    # ------------- CRUD -------------
    def create_session(self, title=DEFAULT_TITLE):
        session_id = uuid.uuid4().hex[:12]
        now = _utcnow()
        payload = {
            "id": session_id,
            "title": title,
            "created": now,
            "updated": now,
            "model": "",
            "messages": [],
        }
        try:
            (self.base / f"{session_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
        self._upsert_index(session_id, title, "")
        return session_id

    def load_session(self, session_id):
        """Returns (messages, meta) with messages API-ready plain dicts."""
        try:
            payload = json.loads(
                (self.base / f"{session_id}.json").read_text(encoding="utf-8")
            )
            messages = payload.get("messages", [])
            if isinstance(messages, list):
                return messages, payload
        except Exception:
            pass
        return None, None

    def save_session(
        self,
        session_id,
        messages,
        title=None,
        model="",
        usage=None,
        design_code=None,
        requirements=None,
    ):
        try:
            path = self.base / f"{session_id}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"id": session_id, "created": _utcnow()}
            payload["messages"] = serialize_messages(messages)
            payload["updated"] = _utcnow()
            if model:
                payload["model"] = model
            if usage is not None:
                payload["usage"] = {
                    "in": int(usage.get("in", 0)),
                    "out": int(usage.get("out", 0)),
                }
            if design_code is not None:
                payload["design_code"] = design_code
            if requirements is not None:
                payload["requirements"] = requirements
            if title is not None:
                payload["title"] = title
            elif not payload.get("title"):
                payload["title"] = autotitle(payload["messages"])
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._upsert_index(
                session_id,
                payload.get("title", DEFAULT_TITLE),
                session_preview(payload["messages"]),
            )
            return True
        except Exception:
            return False

    def rename_session(self, session_id, title):
        """Rename a session without touching its messages. Returns bool."""
        title = " ".join(str(title or "").split())
        if not title or len(title) > 80:
            return False
        try:
            path = self.base / f"{session_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["title"] = title
            payload["updated"] = _utcnow()
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._upsert_index(
                session_id, title, session_preview(payload.get("messages", []))
            )
            return True
        except Exception:
            return False

    def delete_session(self, session_id):
        try:
            (self.base / f"{session_id}.json").unlink()
        except Exception:
            pass
        self._write_index([e for e in self._read_index() if e.get("id") != session_id])
