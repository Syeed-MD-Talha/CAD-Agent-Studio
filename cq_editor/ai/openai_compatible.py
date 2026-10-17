"""OpenAI Chat Completions adapter for CQ-editor's Anthropic-shaped agent loop."""

import json
from types import SimpleNamespace

import requests


class Block(SimpleNamespace):
    def model_dump(self):
        return vars(self).copy()


def _as_dict(block):
    return block if isinstance(block, dict) else block.model_dump()


def _content_parts(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts = []
    for raw in content or []:
        block = _as_dict(raw)
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            source = block["source"]
            url = source.get("url") or f"data:{source['media_type']};base64,{source['data']}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _messages(system, messages):
    converted = [{"role": "system", "content": system}]
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        blocks = [_as_dict(block) for block in content]
        if role == "assistant":
            item = {
                "role": "assistant",
                "content": "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text") or None,
            }
            calls = []
            for block in blocks:
                if block.get("type") == "tool_use":
                    calls.append({"id": block["id"], "type": "function", "function": {
                        "name": block["name"], "arguments": json.dumps(block.get("input", {})),
                    }})
            if calls:
                item["tool_calls"] = calls
            if item["content"] or calls:
                converted.append(item)
            continue
        images = []
        for block in blocks:
            if block.get("type") == "tool_result":
                output = _content_parts(block.get("content", ""))
                text = "\n".join(part["text"] for part in output if part["type"] == "text")
                converted.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": text or "Tool completed."})
                images.extend(part for part in output if part["type"] == "image_url")
            else:
                images.extend(_content_parts([block]))
        if images:
            converted.append({"role": "user", "content": images})
    return converted


def _response(message, finish_reason, usage=None):
    if finish_reason not in ("stop", "length", "tool_calls"):
        raise RuntimeError("Provider did not return a completed reply. No generated code was applied.")
    content = []
    if message.get("reasoning_content"):
        content.append(Block(type="thinking", thinking=message["reasoning_content"]))
    if message.get("content"):
        content.append(Block(type="text", text=message["content"]))
    if finish_reason != "length":
        for call in message.get("tool_calls") or []:
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except (TypeError, ValueError):
                raise RuntimeError("Provider returned incomplete tool arguments. Retry the request.") from None
            if not isinstance(arguments, dict):
                raise RuntimeError("Provider returned tool arguments that are not a JSON object.")
            content.append(Block(type="tool_use", id=call["id"], name=call["function"]["name"], input=arguments))
    usage = usage or {}
    return SimpleNamespace(
        content=content,
        stop_reason={"length": "max_tokens", "tool_calls": "tool_use"}.get(finish_reason, "end_turn"),
        usage=SimpleNamespace(input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0)),
    )


class CompatibleClient:
    """A small client with ``messages.create`` and ``messages.stream`` methods."""

    def __init__(self, api_key, base_url):
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.messages = self

    def _request(self, stream=False, **kwargs):
        payload = {
            "model": kwargs["model"],
            "messages": _messages(kwargs.get("system", ""), kwargs["messages"]),
            "max_tokens": kwargs["max_tokens"],
            "stream": stream,
        }
        if kwargs.get("tools"):
            payload["tools"] = [{"type": "function", "function": {
                "name": tool["name"], "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            }} for tool in kwargs["tools"]]
        try:
            response = requests.post(
                self.base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=payload, stream=stream, timeout=(15, 180),
            )
        except requests.RequestException as error:
            detail = str(error).replace(self._api_key, "[redacted]")[:300]
            raise RuntimeError(f"Could not connect to the provider at {self.base_url}. {detail}") from None
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("error", "")).replace(self._api_key, "[redacted]")[:400]
            except (ValueError, AttributeError):
                detail = ""
            status = response.status_code
            response.close()
            hints = {401: "Check this provider's API key.", 403: "Check API key permissions.", 404: "Check the model ID and API base URL.", 429: "Rate limit or quota reached; retry later."}
            raise RuntimeError(f"Provider HTTP {status}. {hints.get(status, 'Check the provider settings.')} {detail}".strip())
        return response

    def create(self, **kwargs):
        with self._request(**kwargs) as response:
            data = response.json()
        if not data.get("choices"):
            raise RuntimeError("Provider returned no completion choices.")
        choice = data["choices"][0]
        return _response(choice.get("message", {}), choice.get("finish_reason"), data.get("usage"))

    def stream(self, **kwargs):
        return _CompletionStream(self, kwargs)


class _CompletionStream:
    def __init__(self, client, kwargs):
        self.client, self.kwargs, self.final = client, kwargs, None

    def __enter__(self):
        self.response = self.client._request(stream=True, **self.kwargs)
        return self

    def __exit__(self, *args):
        self.response.close()

    def __iter__(self):
        if "application/json" in self.response.headers.get("Content-Type", "").lower():
            data = self.response.json()
            choice = data.get("choices", [{}])[0]
            self.final = _response(choice.get("message", {}), choice.get("finish_reason"), data.get("usage"))
            for block in self.final.content:
                if block.type == "text":
                    yield Block(type="text", text=block.text)
            return
        text, reasoning, calls, finish, usage = [], [], {}, None, {}
        for line in self.response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
                    yield Block(type="text", text=delta["content"])
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if thought:
                    reasoning.append(thought)
                    yield Block(type="thinking", thinking=thought)
                for fragment in delta.get("tool_calls") or []:
                    call = calls.setdefault(fragment["index"], {"id": "", "function": {"name": "", "arguments": ""}})
                    call["id"] += fragment.get("id") or ""
                    function = fragment.get("function") or {}
                    call["function"]["name"] += function.get("name") or ""
                    call["function"]["arguments"] += function.get("arguments") or ""
        if finish is None:
            raise RuntimeError("Provider connection ended before the reply completed. Retry the request.")
        self.final = _response({"content": "".join(text), "reasoning_content": "".join(reasoning), "tool_calls": [calls[i] for i in sorted(calls)]}, finish, usage)

    def get_final_message(self):
        if self.final is None:
            raise RuntimeError("Completion has not finished.")
        return self.final
