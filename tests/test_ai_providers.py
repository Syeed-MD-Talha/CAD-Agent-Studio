from unittest.mock import Mock

import pytest

from cq_editor.ai.openai_compatible import CompatibleClient, _messages, _response
from cq_editor.ai.providers import PROVIDERS, ProviderConfig


def test_provider_profiles_cover_native_and_compatible_apis():
    assert set(PROVIDERS) == {
        "fireworks",
        "featherless",
        "anthropic",
        "custom_openai",
        "custom_anthropic",
    }
    assert PROVIDERS["fireworks"].protocol == "openai"
    assert PROVIDERS["anthropic"].protocol == "anthropic"


def test_provider_config_requires_key_model_and_http_endpoint():
    with pytest.raises(RuntimeError, match="API key"):
        ProviderConfig("fireworks", "", "model", "https://example.com/v1").validate()
    with pytest.raises(RuntimeError, match="model ID"):
        ProviderConfig("fireworks", "key", "", "https://example.com/v1").validate()
    with pytest.raises(RuntimeError, match="base URL"):
        ProviderConfig("fireworks", "key", "model", "not-a-url").validate()


def test_openai_conversion_keeps_tool_results_linked_to_tool_calls():
    converted = _messages(
        "CAD",
        [
            {"role": "user", "content": "Inspect the editor"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "get_editor_code",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "editor text",
                    },
                ],
            },
        ],
    )

    assert converted[-2]["tool_calls"][0]["id"] == "call-1"
    assert converted[-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "editor text",
    }


def test_openai_response_is_normalized_for_existing_tool_dispatch():
    result = _response(
        {
            "content": "I will inspect it.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "get_editor_code", "arguments": "{}"},
                }
            ],
        },
        "tool_calls",
    )

    assert result.stop_reason == "tool_use"
    assert result.content[-1].name == "get_editor_code"
    assert result.content[-1].input == {}


def test_compatible_client_redacts_key_from_http_error(monkeypatch):
    response = Mock(status_code=401)
    response.json.return_value = {"error": "invalid secret-key"}
    response.close = Mock()
    monkeypatch.setattr(
        "cq_editor.ai.openai_compatible.requests.post", Mock(return_value=response)
    )

    with pytest.raises(RuntimeError) as error:
        CompatibleClient("secret-key", "https://example.com/v1").create(
            model="model", system="CAD", messages=[], max_tokens=100
        )

    assert "secret-key" not in str(error.value)
    assert "401" in str(error.value)
    response.close.assert_called_once()
