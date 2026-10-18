"""Injected failures must recover before any incomplete CAD code can execute."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from cq_editor.ai.cad_agent import TOOLS
from cq_editor.ai.openai_compatible import Block, CompatibleClient, _response
from cq_editor.ai.recovery import (
    MAX_RETRIES,
    IncompleteReplyError,
    MalformedToolCallError,
    ProviderError,
    RequestCancelled,
    classify_error,
    complete,
)

SCRIPT = "from build123d import *\nresult = Box(30, 20, 5)\nshow_object(result)"
FENCED = "```python\n" + SCRIPT + "\n```"
PROMPT = [{"role": "user", "content": "Make a 30 x 20 x 5 mm plate"}]


def reply(text="done", stop="end_turn", blocks=None):
    return SimpleNamespace(
        content=blocks if blocks is not None else [Block(type="text", text=text)],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def client_for(*responses):
    return SimpleNamespace(messages=SimpleNamespace(create=Mock(side_effect=responses)))


def run(client, **kwargs):
    return complete(
        client,
        model="test-model",
        system="CAD",
        messages=PROMPT,
        max_tokens=8000,
        tools=TOOLS,
        **kwargs,
    )


@pytest.mark.parametrize(
    "error,expected",
    [
        (ProviderError("busy", 429), "transient"),
        (ProviderError("overloaded", 503), "transient"),
        (ProviderError("context_length_exceeded", 400), "context"),
        (ProviderError("invalid key", 401), "auth"),
        (ProviderError("model missing", 404), "model"),
        (ProviderError("insufficient_quota", 429), "billing"),
        (requests.Timeout(), "transient"),
        (requests.ConnectionError(), "transient"),
        (
            requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
            "transient",
        ),
        (
            requests.exceptions.ContentDecodingError("Incomplete compressed reply"),
            "transient",
        ),
        (requests.exceptions.InvalidURL("invalid endpoint"), "unknown"),
        (ValueError("invalid input"), "unknown"),
    ],
)
def test_classifies_before_retrying(error, expected):
    assert classify_error(error) == expected


@pytest.mark.parametrize(
    "arguments", [{}, {"code": ""}, {"code": "  \n"}, {"code": None}, {"code": 12}, []]
)
def test_empty_cad_call_is_regenerated_once_without_tool_execution(arguments):
    invalid = reply(
        stop="tool_use",
        blocks=[
            Block(
                type="tool_use",
                id="bad-call",
                name="validate_cad_code",
                input=arguments,
            )
        ],
    )
    client = client_for(invalid, reply(FENCED))
    status = []
    original = copy.deepcopy(PROMPT)
    result = run(client, on_status=status.append)
    assert result.content[0].text == FENCED
    assert client.messages.create.call_count == 2
    correction = client.messages.create.call_args.kwargs
    assert "tools" not in correction
    assert "complete" in correction["messages"][-1]["content"].lower()
    assert correction["messages"][0] == PROMPT[0]
    assert "bad-call" not in json.dumps(correction["messages"])
    assert PROMPT == original
    assert "Recovering" in status[0]
    assert result.usage.input_tokens == 20


def test_repeated_empty_calls_stop_after_one_recovery():
    invalid = reply(
        stop="tool_use",
        blocks=[
            Block(
                type="tool_use",
                id="bad",
                name="apply_and_render",
                input={"code": ""},
            )
        ],
    )
    client = client_for(invalid, invalid)
    with pytest.raises(IncompleteReplyError, match="repeatedly returned"):
        run(client)
    assert client.messages.create.call_count == 2


def test_invalid_json_arguments_recover_as_a_complete_script():
    with pytest.raises(MalformedToolCallError) as failure:
        _response(
            {
                "tool_calls": [
                    {
                        "id": "bad",
                        "function": {
                            "name": "validate_cad_code",
                            "arguments": '{"code":',
                        },
                    }
                ]
            },
            "tool_calls",
        )
    result = run(client_for(failure.value, reply(FENCED)))
    assert result.content[0].text == FENCED


def test_valid_tool_call_passes_unchanged():
    response = reply(
        stop="tool_use",
        blocks=[
            Block(
                type="tool_use",
                id="valid",
                name="validate_cad_code",
                input={"code": SCRIPT},
            )
        ],
    )
    client = client_for(response)
    result = run(client)
    assert result.content[0].input["code"] == SCRIPT
    client.messages.create.assert_called_once()


def test_truncated_text_continues_before_returning_a_complete_script():
    split = FENCED.index("result")
    client = client_for(reply(FENCED[:split], "max_tokens"), reply(FENCED[split:]))
    result = run(client)
    assert result.content[0].text == FENCED
    continuation = client.messages.create.call_args.kwargs["messages"]
    assert "Continue exactly" in continuation[-1]["content"]
    assert continuation[0] == PROMPT[0]


def test_partial_tool_json_is_discarded_then_regenerated():
    partial = _response(
        {
            "tool_calls": [
                {
                    "id": "partial",
                    "function": {
                        "name": "apply_and_render",
                        "arguments": '{"code":"unfinished',
                    },
                }
            ]
        },
        "length",
    )
    valid = reply(
        stop="tool_use",
        blocks=[
            Block(
                type="tool_use",
                id="complete",
                name="apply_and_render",
                input={"code": SCRIPT},
            )
        ],
    )
    client = client_for(partial, valid)
    result = run(client)
    assert result.content[0].id == "complete"
    assert "partial" not in json.dumps(
        client.messages.create.call_args.kwargs["messages"][:-1]
    )


def test_continuation_limit_never_returns_partial_code():
    partial = reply("```python\nunfinished", "max_tokens")
    client = client_for(partial, partial, partial)
    with pytest.raises(IncompleteReplyError, match="two recovery"):
        run(client)
    assert client.messages.create.call_count == 3


def test_thinking_exhaustion_has_its_own_bounded_recovery():
    thinking = reply(
        stop="max_tokens", blocks=[Block(type="thinking", thinking="reasoning")]
    )
    client = client_for(thinking, reply(FENCED))
    result = run(client, thinking={"type": "enabled", "budget_tokens": 3000})
    assert result.content[0].text == FENCED
    assert "thinking" not in client.messages.create.call_args.kwargs
    with pytest.raises(IncompleteReplyError, match="after one recovery"):
        run(client_for(thinking, thinking))


def test_unclosed_fence_is_not_returned_as_usable_code():
    with pytest.raises(IncompleteReplyError, match="unfinished code fence"):
        run(client_for(reply("```python\nresult = Box(1, 2, 3)")))


def test_transient_failures_retry_same_request_and_stop_at_limit():
    delays = []
    client = client_for(requests.Timeout(), ProviderError("busy", 429), reply(FENCED))
    assert run(client, sleep=delays.append).content[0].text == FENCED
    assert len(delays) == 2
    assert (
        client.messages.create.call_args_list[0]
        == client.messages.create.call_args_list[2]
    )
    client = client_for(*(ProviderError("busy", 503) for _ in range(MAX_RETRIES + 1)))
    with pytest.raises(ProviderError):
        run(client, sleep=lambda _: None)
    assert client.messages.create.call_count == MAX_RETRIES + 1


@pytest.mark.parametrize(
    "error",
    [
        ProviderError("invalid key for stream", 401),
        ProviderError("unknown model", 404),
        ProviderError("insufficient_quota", 429),
    ],
)
def test_terminal_errors_do_not_retry_or_change_provider(error):
    client = client_for(error)
    with pytest.raises(ProviderError):
        run(client, sleep=lambda _: pytest.fail("must not retry"))
    client.messages.create.assert_called_once()


def test_cancellation_prevents_initial_and_retry_requests():
    client = client_for(reply())
    with pytest.raises(RequestCancelled):
        run(client, cancelled=lambda: True)
    client.messages.create.assert_not_called()
    cancelled = [False]
    client = client_for(ProviderError("busy", 503), reply())
    with pytest.raises(RequestCancelled):
        run(
            client,
            cancelled=lambda: cancelled[0],
            sleep=lambda _: cancelled.__setitem__(0, True),
        )
    client.messages.create.assert_called_once()


def test_context_recovery_preserves_prompts_and_tool_pairing():
    messages = [
        PROMPT[0],
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "old",
                    "name": "validate_cad_code",
                    "input": {"code": SCRIPT * 40},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "old", "content": "error " * 600}
            ],
        },
        {"role": "user", "content": "Keep the dimensions and add four holes."},
    ]
    original = copy.deepcopy(messages)
    client = client_for(
        ProviderError("maximum context length exceeded", 400), reply(FENCED)
    )
    complete(client, model="model", system="CAD", messages=messages, max_tokens=1000)
    first, second = [
        call.kwargs["messages"] for call in client.messages.create.call_args_list
    ]
    assert len(json.dumps(second)) < len(json.dumps(first))
    assert second[0] == messages[0] and second[-1] == messages[-1]
    assert second[1]["content"][0]["id"] == second[2]["content"][0]["tool_use_id"]
    assert messages == original


def test_stream_fallback_only_for_explicit_unsupported_stream():
    client = client_for(reply(FENCED))
    client.messages.stream = Mock(
        side_effect=ProviderError(
            "streaming is not supported", 400, kind="unsupported_stream"
        )
    )
    assert run(client, stream=True).content[0].text == FENCED
    client.messages.create.assert_called_once()
    client.messages.stream = Mock(
        side_effect=ProviderError("invalid key for stream", 401)
    )
    client.messages.create.reset_mock()
    with pytest.raises(ProviderError):
        run(client, stream=True)
    client.messages.create.assert_not_called()


@pytest.mark.parametrize(
    "error_type",
    [
        requests.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ContentDecodingError,
    ],
)
def test_interrupted_stream_resets_preview_and_retries_without_partial_tools(
    error_type,
):
    class Connection:
        def __init__(self, interrupted=False):
            self.interrupted = interrupted

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield Block(
                type="text", text="discarded preview" if self.interrupted else FENCED
            )
            if self.interrupted:
                raise error_type("Response ended prematurely")

        def get_final_message(self):
            return reply(FENCED)

    client = client_for()
    client.messages.stream = Mock(side_effect=[Connection(True), Connection()])
    visible = []
    result = run(
        client,
        stream=True,
        sleep=lambda _: None,
        on_text=visible.append,
        on_reset=visible.clear,
    )
    assert result.content[0].text == FENCED
    assert visible == [FENCED]
    assert client.messages.stream.call_count == 2
    client.messages.create.assert_not_called()


def test_retry_after_is_honored_and_capped(monkeypatch):
    from cq_editor.ai.recovery import backoff_delay

    monkeypatch.setattr("cq_editor.ai.recovery.random.random", lambda: 0)
    assert backoff_delay(0, ProviderError("rate", 429, retry_after="7")) == 7
    assert backoff_delay(0, ProviderError("rate", 429, retry_after="999")) == 30


def test_adapter_stream_error_is_classified_and_redacts_credentials(monkeypatch):
    response = Mock(status_code=200)
    response.headers = {"Content-Type": "text/event-stream"}
    response.iter_lines.return_value = [
        'data: {"error":{"status":503,"message":"busy test-secret"}}'
    ]
    monkeypatch.setattr(
        "cq_editor.ai.openai_compatible.requests.post", Mock(return_value=response)
    )
    with pytest.raises(ProviderError) as failure:
        with CompatibleClient("test-secret", "https://example.com/v1").stream(
            model="model", messages=PROMPT, max_tokens=100
        ) as connection:
            list(connection)
    assert classify_error(failure.value) == "transient"
    assert "test-secret" not in str(failure.value)
    response.close.assert_called_once()


@pytest.mark.parametrize("exhausted", [False, True])
def test_chunked_stream_failure_discards_partial_tools_and_closes_each_response(
    monkeypatch, exhausted
):
    def interrupted_lines():
        yield "data: " + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "discarded preview",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "partial",
                                    "function": {
                                        "name": "apply_and_render",
                                        "arguments": '{"code":"unfinished',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")

    responses = []
    for _ in range(MAX_RETRIES + 1 if exhausted else 1):
        response = Mock(status_code=200)
        response.headers = {"Content-Type": "text/event-stream"}
        response.iter_lines.return_value = interrupted_lines()
        responses.append(response)
    if not exhausted:
        response = Mock(status_code=200)
        response.headers = {"Content-Type": "text/event-stream"}
        response.iter_lines.return_value = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "complete",
                                        "function": {
                                            "name": "apply_and_render",
                                            "arguments": json.dumps({"code": SCRIPT}),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        responses.append(response)
    post = Mock(side_effect=responses)
    monkeypatch.setattr("cq_editor.ai.openai_compatible.requests.post", post)
    client = CompatibleClient("test-secret", "https://example.com/v1")
    visible, statuses = [], []
    options = dict(
        stream=True,
        sleep=lambda _: None,
        on_text=visible.append,
        on_reset=visible.clear,
        on_status=statuses.append,
    )
    if exhausted:
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            run(client, **options)
    else:
        result = run(client, **options)
        assert len(result.content) == 1
        assert result.content[0].id == "complete"
        assert result.content[0].input == {"code": SCRIPT}
        assert visible == []
    assert post.call_count == len(responses)
    assert len(statuses) == (MAX_RETRIES if exhausted else 1)
    assert all(call == post.call_args_list[0] for call in post.call_args_list)
    for response in responses:
        response.close.assert_called_once()


def test_worker_recovers_before_emitting_a_response(qtbot):
    from cq_editor.widgets.chat import LLMWorker

    invalid = reply(
        stop="tool_use",
        blocks=[
            Block(
                type="tool_use",
                id="bad",
                name="validate_cad_code",
                input={},
            )
        ],
    )
    client = client_for(invalid, reply(FENCED))
    worker = LLMWorker(client, "model", PROMPT)
    with qtbot.waitSignal(worker.finished_ok, timeout=2000) as signal:
        worker.start()
    worker.wait(2000)
    assert signal.args[0].content[0].text == FENCED
    assert client.messages.create.call_count == 2


@pytest.fixture
def chat(qtbot, tmp_path, monkeypatch):
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QWidget
    from cq_editor.ai.sessions import SessionStore
    from cq_editor.widgets import chat as module

    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setattr(
        module, "SessionStore", lambda: SessionStore(tmp_path / "sessions")
    )
    from cq_editor.ai.skill_store import SkillStore

    monkeypatch.setattr(module, "SkillStore", lambda: SkillStore(tmp_path / "skills"))
    parent = QWidget()
    parent.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    qtbot.addWidget(parent)
    widget = module.ChatWidget(parent)
    widget._test_parent = parent
    qtbot.addWidget(widget)
    return widget


@pytest.mark.parametrize("name", ["validate_cad_code", "apply_and_render"])
def test_dispatch_never_passes_empty_code_to_cad(chat, monkeypatch, name):
    validate = Mock()
    apply = Mock()
    monkeypatch.setattr(chat, "_validate_isolated", validate)
    monkeypatch.setattr(chat, "apply_code", apply)
    chat.last_code = SCRIPT
    for arguments in ({}, {"code": "\n "}, {"code": None}, {"code": []}):
        assert "non-empty string" in chat.execute_tool(name, arguments)
    validate.assert_not_called()
    apply.assert_not_called()
    assert chat.last_code == SCRIPT


def test_failed_validation_never_writes_editor_and_can_self_fix(chat, monkeypatch):
    apply = Mock(return_value=(True, "OK"))
    monkeypatch.setattr(
        chat, "_validate_isolated", lambda _: "Traceback: invalid geometry"
    )
    monkeypatch.setattr(chat, "_set_editor_and_render", apply)
    monkeypatch.setattr(chat, "_kick_llm", Mock())
    chat.iters_left = 6
    chat.on_llm_ok(reply(FENCED))
    apply.assert_not_called()
    chat._kick_llm.assert_called_once()
    assert "invalid geometry" in chat.messages[-1]["content"]


def test_ui_rejects_partial_reply_even_if_worker_is_bypassed(chat, monkeypatch):
    apply = Mock()
    monkeypatch.setattr(chat, "apply_code", apply)
    chat.iters_left = 6
    chat.on_llm_ok(reply(FENCED, "max_tokens"))
    apply.assert_not_called()
    assert "incomplete" in chat.status.text()


def test_tool_call_with_code_in_prose_does_not_apply_twice(chat, monkeypatch):
    apply = Mock(return_value="OK: rendered")
    monkeypatch.setattr(chat, "apply_code", apply)
    monkeypatch.setattr(chat, "_kick_llm", Mock())
    chat.iters_left = 6
    chat.autofix.setChecked(False)
    chat.vision_check.setChecked(False)
    chat.on_llm_ok(
        reply(
            FENCED,
            "tool_use",
            blocks=[
                Block(type="text", text=FENCED),
                Block(
                    type="tool_use",
                    id="apply",
                    name="apply_and_render",
                    input={"code": SCRIPT},
                ),
            ],
        )
    )
    apply.assert_called_once_with(SCRIPT)


def tool_reply(name, code=None):
    return reply(
        stop="tool_use",
        blocks=[
            Block(
                type="tool_use",
                id=name,
                name=name,
                input={"code": code} if code is not None else {},
            )
        ],
    )


def test_successful_tools_continue_with_self_fix_off(chat, monkeypatch):
    kick = Mock()
    monkeypatch.setattr(chat, "_kick_llm", kick)
    monkeypatch.setattr(
        chat,
        "execute_tool",
        Mock(
            side_effect=[
                "(editor empty)",
                "OK: validation passed",
                "OK: rendered",
            ]
        ),
    )
    chat.iters_left = 6
    chat.autofix.setChecked(False)
    chat.vision_check.setChecked(False)
    for name in ("get_editor_code", "validate_cad_code", "apply_and_render"):
        chat.on_llm_ok(tool_reply(name, SCRIPT if name != "get_editor_code" else None))
        assert chat.messages[-1]["content"][0]["is_error"] is False
    assert kick.call_count == 3
    assert chat._request_rendered
    chat.on_llm_ok(reply("The plate has been rendered."))
    assert chat.status.text() == "Model rendered."
    assert kick.call_count == 3


def test_failed_tool_stops_only_when_self_fix_is_off(chat, monkeypatch):
    kick = Mock()
    monkeypatch.setattr(chat, "_kick_llm", kick)
    monkeypatch.setattr(
        chat, "execute_tool", Mock(return_value="Traceback: missing Cylinder")
    )
    chat.iters_left = 6
    chat.autofix.setChecked(False)
    chat.on_llm_ok(tool_reply("validate_cad_code", SCRIPT))
    kick.assert_not_called()
    assert "Automatic repair is off" in chat.status.text()
    assert "missing Cylinder" in chat.status.text()
    assert chat.messages[-1]["content"][0]["is_error"]


def test_tool_limit_gets_one_final_script_then_stops(chat, monkeypatch):
    kick = Mock()
    apply = Mock(return_value="OK: rendered")
    monkeypatch.setattr(chat, "_kick_llm", kick)
    monkeypatch.setattr(
        chat, "execute_tool", Mock(return_value="OK: validation passed")
    )
    monkeypatch.setattr(chat, "apply_code", apply)
    chat.iters_left = 6
    for _ in range(6):
        chat.on_llm_ok(tool_reply("validate_cad_code", SCRIPT))
    assert chat.iters_left == 0
    assert chat._finalizing
    assert "Do not call any more tools" in chat.messages[-1]["content"]
    assert kick.call_count == 6
    chat.on_llm_ok(reply(FENCED))
    apply.assert_called_once_with(SCRIPT)
    assert chat.status.text() == "Model rendered."
    assert kick.call_count == 6


def test_final_attempt_will_not_run_more_tools(chat, monkeypatch):
    execute = Mock()
    monkeypatch.setattr(chat, "execute_tool", execute)
    monkeypatch.setattr(chat, "_kick_llm", Mock())
    chat.iters_left = 0
    chat._finalizing = True
    chat.on_llm_ok(tool_reply("apply_and_render", SCRIPT))
    execute.assert_not_called()
    chat._kick_llm.assert_not_called()
    assert "after the final attempt" in chat.status.text()
    assert chat.messages[-1]["content"][0]["is_error"]


def test_final_validation_failure_does_not_restart_tool_loop(chat, monkeypatch):
    monkeypatch.setattr(
        chat, "_validate_isolated", lambda _: "Traceback: unsupported operation"
    )
    monkeypatch.setattr(chat, "_kick_llm", Mock())
    chat._finalizing = True
    chat.iters_left = 0
    chat.on_llm_ok(reply(FENCED))
    chat._kick_llm.assert_not_called()
    assert "final attempt failed" in chat.status.text()
    assert "unsupported operation" in chat.status.text()


def test_failed_checks_status_acknowledges_model_already_displayed(chat, monkeypatch):
    monkeypatch.setattr(
        chat,
        "apply_code",
        Mock(
            return_value="Error: design verification failed; model displayed for review.\nFailed: Width"
        ),
    )
    chat._finalizing = True
    chat.iters_left = 0
    chat.on_llm_ok(reply(FENCED))
    assert "Model displayed for review" in chat.status.text()
    assert "Code was not applied" not in chat.status.text()


def test_stop_cancels_final_attempt(chat, monkeypatch):
    monkeypatch.setattr(chat, "apply_code", Mock())
    chat._finalizing = True
    chat.iters_left = 0
    chat.on_stop()
    chat.on_llm_ok(reply(FENCED))
    chat.apply_code.assert_not_called()
    assert "Stopped" in chat.status.text()


def test_final_worker_omits_tools_from_request(qtbot):
    from cq_editor.widgets.chat import LLMWorker

    client = client_for(reply(FENCED))
    worker = LLMWorker(client, "model", PROMPT, tools=[])
    with qtbot.waitSignal(worker.finished_ok, timeout=2000):
        worker.start()
    worker.wait(2000)
    assert "tools" not in client.messages.create.call_args.kwargs


def test_limit_after_successful_render_requests_summary_without_repeating_apply(
    chat, monkeypatch
):
    apply = Mock(return_value="OK: rendered")
    kick = Mock()
    monkeypatch.setattr(chat, "apply_code", apply)
    monkeypatch.setattr(chat, "_kick_llm", kick)
    chat.vision_check.setChecked(False)
    chat.iters_left = 1
    chat.on_llm_ok(tool_reply("apply_and_render", SCRIPT))
    assert chat._finalizing
    assert "Summarize the actual result" in chat.messages[-1]["content"]
    chat.on_llm_ok(reply("The model was rendered successfully."))
    apply.assert_called_once_with(SCRIPT)
    kick.assert_called_once()
    assert chat.status.text() == "Model rendered."


def test_auto_apply_preference_is_independent_for_each_chat(chat, qtbot):
    from cq_editor.widgets.chat import ChatWidget

    other = ChatWidget(chat._test_parent)
    qtbot.addWidget(other)
    chat.preferences["Auto-apply code"] = False
    assert other.preferences["Auto-apply code"] is True


def test_paste_image_shortcut_attaches_a_reference_image(chat, qtbot):
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QColor, QImage, QKeyEvent
    from PyQt5.QtWidgets import QApplication

    image = QImage(40, 20, QImage.Format_RGB32)
    image.fill(QColor("slategray"))
    QApplication.clipboard().setImage(image)
    chat.input.setFocus()
    QApplication.sendEvent(
        chat.input,
        QKeyEvent(QEvent.KeyPress, Qt.Key_V, Qt.ControlModifier),
    )

    assert len(chat.pending_images) == 1
    assert chat.pending_images[0]["name"] == "Pasted image"
    assert chat.pending_images[0]["w"] == 40
    assert not chat.preview_bar.isHidden()
    QApplication.clipboard().clear()


def test_paste_text_shortcut_keeps_normal_text_pasting(chat):
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtWidgets import QApplication

    QApplication.clipboard().setText("Create a mounting plate")
    chat.input.setFocus()
    QApplication.sendEvent(
        chat.input,
        QKeyEvent(QEvent.KeyPress, Qt.Key_V, Qt.ControlModifier),
    )

    assert chat.input.toPlainText() == "Create a mounting plate"
    assert not chat.pending_images
    QApplication.clipboard().clear()


def test_requirements_cannot_be_weakened_during_repair(chat):
    initial = {
        "requirements": [
            dict(
                kind="extent",
                expected=30,
                axis="X",
                source="30 mm long",
                label="Length",
            )
        ]
    }
    assert chat.execute_tool("set_design_requirements", initial).startswith("OK")
    relaxed = copy.deepcopy(initial)
    relaxed["requirements"][0]["expected"] = 25
    assert chat.execute_tool("set_design_requirements", relaxed).startswith("Error:")
    assert chat.design_requirements[0]["expected"] == 30


def test_multi_view_capture_restores_exact_camera_and_labels_images(chat, monkeypatch):
    from OCP.Graphic3d import Graphic3d_Camera
    from OCP.gp import gp_Dir

    camera = Graphic3d_Camera()
    before = tuple(camera.Direction().Coord())

    class View:
        def Camera(self):
            return camera

        def SetProj(self, orientation):
            camera.SetDirection(gp_Dir(1, 2, 3))

        def FitAll(self):
            pass

        def Redraw(self):
            pass

    main = SimpleNamespace(
        components={"viewer": SimpleNamespace(_get_view=lambda: View())}
    )
    monkeypatch.setattr(chat, "_main", lambda: main)
    monkeypatch.setattr(chat, "_capture_screenshot", lambda: b"jpeg")
    monkeypatch.setattr(chat, "_editor_text", lambda: SCRIPT)
    chat.verification_report = {"rows": [], "ok": True}
    views = chat._capture_design_views()
    assert [label for label, data in views] == ["Isometric", "Front", "Top", "Side"]
    assert tuple(camera.Direction().Coord()) == before
    blocks = chat._view_blocks(views)
    assert len([block for block in blocks if block["type"] == "image"]) == 4
    assert chat.execute_tool(
        "record_visual_review",
        {"status": "Passed", "notes": "Inspected all four views; shape matches."},
    ).startswith("OK")
    chat._on_editor_changed()
    assert chat.execute_tool(
        "record_visual_review", {"status": "Passed", "notes": "stale"}
    ).startswith("Error")


def test_stop_retains_network_thread_and_discards_its_late_reply(
    chat, qtbot, monkeypatch
):
    import time
    from cq_editor.widgets.chat import LLMWorker

    def delayed(**kwargs):
        time.sleep(0.15)
        return reply(FENCED)

    client = SimpleNamespace(messages=SimpleNamespace(create=delayed))
    worker = LLMWorker(client, "model", PROMPT, chat)
    chat.worker = worker
    chat.iters_left = 6
    apply = Mock()
    monkeypatch.setattr(chat, "apply_code", apply)
    worker.finished_ok.connect(chat.on_llm_ok)
    worker.finished_err.connect(chat.on_llm_err)
    chat._retain_worker(worker)
    worker.start()
    chat.on_stop()
    assert chat.worker is None and chat.send_btn.isEnabled()
    assert worker in chat._active_workers
    qtbot.waitUntil(lambda: worker not in chat._active_workers, timeout=3000)
    apply.assert_not_called()


def test_stream_updates_replace_only_live_message_and_bound_preview(chat, monkeypatch):
    chat._append("You", "Original request stays visible")
    rebuild = Mock(wraps=chat.history.setHtml)
    monkeypatch.setattr(chat.history, "setHtml", rebuild)
    chat._append_stream_start()
    for index in range(20):
        chat._stream_buf += f"chunk {index}: " + "x" * 1500
        chat._refresh_stream_slot()
    visible = chat.history.toPlainText()
    assert "Original request stays visible" in visible
    assert visible.count("Assistant > live") == 1
    assert "chunk 19" in visible
    assert len(visible) < 20000
    rebuild.assert_not_called()
    full_response = chat._stream_buf
    chat._finish_stream_display()
    assert "chunk 0" in chat.history.toPlainText()
    assert chat._stream_buf == full_response
    chat._append("System", "Completed")
    assert "Original request stays visible" in chat.history.toPlainText()
    assert "Completed" in chat.history.toPlainText()


def test_theme_change_keeps_inflight_stream_buffers(chat):
    chat._append_stream_start()
    chat._stream_buf = "Partial reply"
    chat._think_buf = "Partial reasoning"
    chat._on_app_theme_changed()
    assert chat._stream_active
    assert chat._stream_buf == "Partial reply"
    assert chat._think_buf == "Partial reasoning"
    assert "Partial reply" in chat.history.toPlainText()
    chat._finish_stream_display()


def test_example_tool_reads_source_without_requiring_a_main_window(chat):
    output = chat.execute_tool("get_examples", {"name": "09_threaded_bolt_and_nut"})
    assert "def threaded_shaft" in output
    assert "show_object(bolt" in output


def test_subagent_can_actually_read_bundled_examples():
    from cq_editor.ai.subagent import run_subagent

    client = client_for(tool_reply("get_examples"), reply("Source inspected."))
    client.messages.create.side_effect = [
        reply(
            stop="tool_use",
            blocks=[
                Block(
                    type="tool_use",
                    id="example",
                    name="get_examples",
                    input={"name": "06_sweep"},
                )
            ],
        ),
        reply("Source inspected."),
    ]
    result = run_subagent("Read the sweep example", client=client, model="model")
    assert result["ok"]
    calls = client.messages.create.call_args_list
    assert {tool["name"] for tool in calls[0].kwargs["tools"]} == {
        "get_examples",
        "validate_cad_code",
        "skill_view",
    }
    assert "JernArc" in calls[1].kwargs["messages"][-1]["content"][0]["content"]


def test_chat_skill_read_returns_full_instructions_as_a_tool_result(chat, monkeypatch):
    monkeypatch.setattr(chat, "_kick_llm", Mock())
    chat.iters_left = 6
    response = reply(
        stop="tool_use",
        blocks=[
            Block(
                type="tool_use",
                id="read-skill",
                name="skill_view",
                input={"name": "threaded-fasteners"},
            )
        ],
    )
    chat.on_llm_ok(response)
    output = chat.messages[-1]["content"][0]
    assert output["tool_use_id"] == "read-skill" and not output["is_error"]
    assert "09_threaded_bolt_and_nut" in output["content"]


def test_chat_can_save_and_activate_learned_skills(chat):
    chat._skills.validator = Mock(return_value="OK: one valid object")
    arguments = {
        "action": "save_draft",
        "name": "fixture-plates",
        "description": "Create fixture plates with sensible parameter bounds.",
        "instructions": "Keep the requested plate dimensions parametric and inspect hole clearances before validation.",
        "example_code": SCRIPT,
    }
    assert chat.execute_tool("skill_manage", arguments).startswith("OK:")
    assert "fixture-plates:" not in chat._skills.index()
    assert chat.execute_tool(
        "skill_manage", {"action": "activate", "name": "fixture-plates"}
    ).startswith("OK:")
    assert "fixture-plates:" in chat._skills.index()
    assert not chat._request_rendered


@pytest.mark.parametrize("plan,learning", [(True, True), (False, False)])
def test_skill_writes_disabled_in_plan_mode_or_preferences(chat, plan, learning):
    chat.plan_mode = plan
    chat.preferences["Learn reusable CAD skills"] = learning
    assert "disabled" in chat.execute_tool(
        "skill_manage", {"action": "activate", "name": "fixture-plates"}
    )
    assert chat.execute_tool("skill_view", {"name": "assemblies"}).startswith(
        "Skill assemblies"
    )
    assert not chat._skills.root.exists()


def test_worker_sends_only_skill_descriptions(qtbot, tmp_path):
    from cq_editor.ai.skill_store import SkillStore
    from cq_editor.widgets.chat import LLMWorker

    client = client_for(reply("Ready."))
    worker = LLMWorker(client, "model", PROMPT, skills=SkillStore(tmp_path / "skills"))
    with qtbot.waitSignal(worker.finished_ok, timeout=2000):
        worker.start()
    worker.wait(2000)
    system = client.messages.create.call_args.kwargs["system"]
    assert "threaded-fasteners:" in system
    assert "overlap the ridge root slightly" not in system


def test_disabled_learning_is_not_advertised_to_model(chat, monkeypatch):
    from cq_editor.widgets import chat as module

    worker = Mock()
    constructor = Mock(return_value=worker)
    monkeypatch.setattr(module, "LLMWorker", constructor)
    chat.stream_check.setChecked(False)
    chat.preferences["Learn reusable CAD skills"] = False
    chat._model, chat._client = "model", object()
    chat._kick_llm()
    keywords = constructor.call_args.kwargs
    assert "skill_manage" not in {tool["name"] for tool in keywords["tools"]}
    assert "skill_view" in {tool["name"] for tool in keywords["tools"]}
    assert keywords["learning"] is False
    chat._set_busy(False)


def test_subagent_reads_skill_instructions_without_write_tools(tmp_path):
    from cq_editor.ai.skill_store import SkillStore
    from cq_editor.ai.subagent import run_subagent

    client = client_for(
        reply(
            stop="tool_use",
            blocks=[
                Block(
                    type="tool_use",
                    id="skill",
                    name="skill_view",
                    input={"name": "geometry-repair"},
                )
            ],
        ),
        reply("Use the global extrude function."),
    )
    result = run_subagent(
        "Repair a failing sketch extrusion",
        client=client,
        model="model",
        skills=SkillStore(tmp_path / "skills"),
    )
    assert result["ok"]
    calls = client.messages.create.call_args_list
    assert "skill_manage" not in {tool["name"] for tool in calls[0].kwargs["tools"]}
    assert (
        "extrude(sketch, amount=height)"
        in calls[1].kwargs["messages"][-1]["content"][0]["content"]
    )
