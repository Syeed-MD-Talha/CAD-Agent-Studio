"""Bounded provider recovery; no CAD tools execute until a reply is complete."""

import json
import random
import time
from types import SimpleNamespace

import requests

from .sessions import serialize_messages

MAX_RETRIES = 2
MAX_CONTINUATIONS = 2


class ProviderError(RuntimeError):
    def __init__(self, message, status_code=None, retry_after=None, kind=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.kind = kind


class IncompleteReplyError(RuntimeError):
    pass


class MalformedToolCallError(IncompleteReplyError):
    pass


class RequestCancelled(RuntimeError):
    pass


class Block(SimpleNamespace):
    def model_dump(self):
        return vars(self).copy()


def classify_error(error):
    """Classify before retrying; invalid keys, models and quota do not retry."""
    if isinstance(error, MalformedToolCallError):
        return "malformed_tool"
    if getattr(error, "kind", None):
        return error.kind
    status = getattr(error, "status_code", None)
    text = str(error).lower()
    if any(
        term in text
        for term in (
            "insufficient_quota",
            "quota exhausted",
            "insufficient credits",
            "credit balance",
            "billing hard limit",
            "payment required",
        )
    ):
        return "billing"
    if status in (400, 413, 422) and any(
        term in text
        for term in (
            "context length",
            "context window",
            "context_length",
            "context limit",
            "maximum context",
            "prompt is too long",
            "prompt too long",
            "too many tokens",
            "input is too long",
            "input tokens exceed",
        )
    ):
        return "context"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "model"
    if status in (408, 409, 429) or isinstance(status, int) and 500 <= status < 600:
        return "transient"
    if isinstance(
        error,
        (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ContentDecodingError,
        ),
    ) or type(error).__name__ in (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    ):
        return "transient"
    return "unknown"


def backoff_delay(attempt, error):
    explicit = getattr(error, "retry_after", None)
    if explicit is None:
        headers = getattr(getattr(error, "response", None), "headers", {})
        explicit = headers.get("retry-after") if headers else None
    try:
        explicit = float(explicit)
    except (TypeError, ValueError):
        explicit = 0
    base = min(8, 2**attempt)
    return min(30, max(explicit, base + random.random() * base * 0.5))


def cad_argument_error(name, arguments):
    if name not in ("validate_cad_code", "apply_and_render"):
        return None
    code = arguments.get("code") if isinstance(arguments, dict) else None
    if not isinstance(code, str) or not code.strip():
        return (
            f"Invalid {name} call: 'code' must be a non-empty string containing the "
            "complete Python script. No code was executed. Resend the complete script; "
            "do not send an empty object, null, or whitespace."
        )
    return None


def _plain(messages):
    return serialize_messages(messages)


def _prompt(message):
    if message.get("role") != "user":
        return False
    content = message.get("content")
    return (
        isinstance(content, str)
        or isinstance(content, list)
        and any(
            block.get("type") in ("text", "image")
            for block in content
            if isinstance(block, dict)
        )
    )


def compact_for_request(messages, force=False):
    """Shorten historical tool/reply data, retaining prompts and live exchanges."""
    from .cad_agent import compact_messages

    original = _plain(messages)
    payload = compact_messages(original)
    keep_from = max(
        (i for i, message in enumerate(original) if _prompt(message)), default=0
    )
    for index, message in enumerate(original):
        if _prompt(message) or index >= keep_from:
            payload[index] = message
        elif force:
            # Only old outputs shrink further; IDs and tool/result order survive.
            copied = json.loads(json.dumps(payload[index]))
            content = copied.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        block["text"] = block.get("text", "")[:300]
                    elif block.get("type") == "tool_use":
                        block["input"] = {
                            key: value[:200] if isinstance(value, str) else value
                            for key, value in (block.get("input") or {}).items()
                        }
                    elif block.get("type") == "tool_result":
                        result = block.get("content")
                        if isinstance(result, str):
                            block["content"] = result[:300]
                        elif isinstance(result, list):
                            block["content"] = [
                                {
                                    "type": "text",
                                    "text": "[Earlier tool output omitted]",
                                }
                            ]
            payload[index] = copied
    return payload


def _join_text(prefix, suffix):
    for size in range(min(len(prefix), len(suffix), 2000), 7, -1):
        if prefix.endswith(suffix[:size]):
            return prefix + suffix[size:]
    return prefix + suffix


def complete(
    client,
    *,
    model,
    system,
    messages,
    max_tokens,
    tools=None,
    thinking=None,
    stream=False,
    on_text=None,
    on_thinking=None,
    on_reset=None,
    on_status=None,
    cancelled=lambda: False,
    sleep=None,
):
    """Return complete replies, with independent finite recovery budgets.

    Malformed CAD arguments are regenerated once without tools. Text truncation
    continues up to twice. Partial tool JSON is discarded and regenerated.
    """
    on_status = on_status or (lambda text: None)
    totals = {"input_tokens": 0, "output_tokens": 0}
    base = compact_for_request(messages)
    extra, prefix = [], ""
    continuations = malformed = reasoning_retries = context_retries = retries = 0
    script_required = False
    streaming = stream

    def check_cancelled():
        if cancelled():
            raise RequestCancelled("Stopped by user.")

    def wait(delay):
        if sleep is not None:
            sleep(delay)
        else:
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                check_cancelled()
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        check_cancelled()

    def reset():
        if on_reset:
            on_reset()
        if on_reset and prefix and on_text:
            on_text(prefix)

    while True:
        check_cancelled()
        kwargs = dict(
            model=model, system=system, messages=base + extra, max_tokens=max_tokens
        )
        if tools and not script_required:
            kwargs["tools"] = tools
        if thinking is not None:
            kwargs["thinking"] = thinking
        try:
            if streaming:
                if not callable(getattr(client.messages, "stream", None)):
                    streaming = False
                    reset()
                    continue
                try:
                    with client.messages.stream(**kwargs) as connection:
                        for event in connection:
                            check_cancelled()
                            if event.type == "text" and on_text:
                                on_text(event.text)
                            elif event.type == "thinking" and on_thinking:
                                on_thinking(event.thinking)
                        response = connection.get_final_message()
                except NotImplementedError:
                    streaming = False
                    reset()
                    continue
            else:
                response = client.messages.create(**kwargs)
        except Exception as error:
            check_cancelled()
            kind = classify_error(error)
            if kind == "unsupported_stream" and streaming:
                streaming = False
                on_status(
                    "Provider does not support streaming; requesting a complete reply"
                )
                reset()
                continue
            if kind == "context" and context_retries < 1:
                context_retries += 1
                compacted = compact_for_request(messages, force=True)
                if compacted == base:
                    raise IncompleteReplyError(
                        "The current request exceeds this model's context limit. Shorten the "
                        "request or choose a model with a larger context window. No code was applied."
                    ) from None
                base = compacted
                on_status(
                    "Context limit reached. Shortening older tool outputs and retrying"
                )
                reset()
                continue
            if kind == "transient" and retries < MAX_RETRIES:
                delay = backoff_delay(retries, error)
                retries += 1
                on_status(
                    f"Temporary provider failure. Retry {retries}/{MAX_RETRIES} in {delay:.0f}s"
                )
                reset()
                wait(delay)
                continue
            if kind != "malformed_tool":
                raise
            response = SimpleNamespace(
                content=[], stop_reason="malformed_tool", usage=None
            )

        check_cancelled()
        usage = getattr(response, "usage", None)
        for key in totals:
            totals[key] += int(getattr(usage, key, 0) or 0)
        content = _plain([{"role": "assistant", "content": response.content}])[0][
            "content"
        ]
        calls = [block for block in content if block.get("type") == "tool_use"]
        text = "".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
        stop = getattr(response, "stop_reason", "")
        truncated = stop in ("max_tokens", "length")
        invalid = stop == "malformed_tool" or any(
            cad_argument_error(block.get("name"), block.get("input")) for block in calls
        )
        if invalid and not truncated:
            if malformed:
                raise IncompleteReplyError(
                    "The model repeatedly returned a CAD tool call without valid code. "
                    "Automatic recovery stopped. No incomplete code was applied; try another model."
                )
            malformed += 1
            script_required = True
            prefix = ""
            extra = [
                {
                    "role": "user",
                    "content": (
                        "Your CAD tool call had missing, empty, or invalid code arguments and was not "
                        "executed. Return the COMPLETE requested design as ONE ```python fenced script, "
                        "ending with show_object(result). Do not call tools. Preserve the original requirements."
                    ),
                }
            ]
            on_status(
                "Empty or invalid CAD arguments. Recovering with a complete Python script"
            )
            reset()
            continue
        if truncated:
            if (
                not text
                and not calls
                and not getattr(response, "incomplete_tools", False)
            ):
                if reasoning_retries:
                    raise IncompleteReplyError(
                        "The model exhausted its output on reasoning after one recovery. "
                        "Choose lower effort or another model. No code was applied."
                    )
                reasoning_retries += 1
                thinking = None
                on_status(
                    "Output was exhausted before a reply. Retrying without a thinking budget"
                )
                reset()
                continue
            if continuations >= MAX_CONTINUATIONS:
                raise IncompleteReplyError(
                    "Reply is still truncated after two recovery attempts. No partial code was applied."
                )
            continuations += 1
            if calls or getattr(response, "incomplete_tools", False):
                prefix = ""
                extra = [
                    {
                        "role": "user",
                        "content": (
                            "Your previous tool call was truncated and NOT executed. Resend the complete "
                            "tool call including the full non-empty code string. Do not continue partial JSON."
                        ),
                    }
                ]
                on_status("Truncated tool call. Regenerating complete arguments")
                reset()
            else:
                prefix = _join_text(prefix, text)
                extra = [
                    {"role": "assistant", "content": prefix},
                    {
                        "role": "user",
                        "content": (
                            "Continue exactly from where the response was cut off. Do not restart, repeat, "
                            "or summarize. Finish the Python script and close its code fence."
                        ),
                    },
                ]
                on_status("Reply truncated. Continuing the complete script")
            continue
        if prefix:
            if calls:
                raise IncompleteReplyError(
                    "The model switched to tools during a text continuation. No partial code was applied."
                )
            text = _join_text(prefix, text)
            content = [{"type": "text", "text": text}]
            if on_reset:
                on_reset()
                if on_text:
                    on_text(text)
        if script_required and calls:
            raise IncompleteReplyError(
                "The model did not return a complete script during recovery. No code was applied."
            )
        if text.count("```") % 2:
            raise IncompleteReplyError(
                "The model returned an unfinished code fence. No partial code was applied."
            )
        if script_required:
            from .cad_agent import extract_python_code

            if not extract_python_code(text).strip():
                raise IncompleteReplyError(
                    "Recovery returned no complete Python script. No code was applied."
                )
        if not text.strip() and not calls:
            raise IncompleteReplyError(
                "The model returned no text or code. Try another model or resend your request."
            )
        return SimpleNamespace(
            content=[Block(**block) for block in content],
            stop_reason=stop,
            usage=SimpleNamespace(**totals),
        )
