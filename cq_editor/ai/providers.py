"""Provider profiles and client construction for CQ-editor AI Chat."""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Provider:
    label: str
    protocol: str
    default_base_url: str
    key_env: str
    model_env: str
    default_model: str
    base_env: str


PROVIDERS = {
    "fireworks": Provider(
        "Fireworks", "openai", "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY", "FIREWORKS_MODEL_ID",
        "accounts/fireworks/models/kimi-k2-instruct-0905", "FIREWORKS_BASE_URL",
    ),
    "featherless": Provider(
        "Featherless", "openai", "https://api.featherless.ai/v1",
        "FEATHERLESS_API_KEY", "FEATHERLESS_MODEL_ID",
        "Qwen/Qwen3-32B", "FEATHERLESS_BASE_URL",
    ),
    "anthropic": Provider(
        "Anthropic (Claude)", "anthropic", "https://api.anthropic.com",
        "ANTHROPIC_API_KEY", "MODEL_ID", "claude-sonnet-4-6", "ANTHROPIC_BASE_URL",
    ),
    "custom_openai": Provider(
        "Custom OpenAI-compatible", "openai", "", "OPENAI_API_KEY",
        "OPENAI_MODEL_ID", "", "OPENAI_API_BASE",
    ),
    "custom_anthropic": Provider(
        "Custom Anthropic-compatible", "anthropic", "", "CUSTOM_ANTHROPIC_API_KEY",
        "CUSTOM_ANTHROPIC_MODEL_ID", "", "CUSTOM_ANTHROPIC_BASE_URL",
    ),
}


@dataclass
class ProviderConfig:
    provider: str
    api_key: str = field(default="", repr=False)
    model: str = ""
    base_url: str = ""

    def validate(self):
        if self.provider not in PROVIDERS:
            raise RuntimeError("Choose a supported AI provider.")
        spec = PROVIDERS[self.provider]
        if not self.api_key.strip():
            raise RuntimeError(
                f"No API key for {spec.label}. Open Provider settings or set {spec.key_env} in .env."
            )
        if not self.model.strip():
            raise RuntimeError(f"Enter a model ID for {spec.label} in Provider settings.")
        parsed = urlsplit(self.base_url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise RuntimeError("Enter an API base URL starting with https:// or http://.")


def load_environment():
    """Load a local .env without overriding the process environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (Path(__file__).resolve().parents[2] / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return
    load_dotenv(override=False)


def default_provider():
    explicit = os.getenv("CQ_AI_PROVIDER", "").strip().lower()
    if explicit in PROVIDERS:
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    for provider in ("fireworks", "featherless", "custom_openai", "custom_anthropic"):
        if os.getenv(PROVIDERS[provider].key_env):
            return provider
    return "fireworks"


def environment_config(provider):
    if provider not in PROVIDERS:
        raise RuntimeError("Choose a supported AI provider.")
    spec = PROVIDERS[provider]
    model = os.getenv(spec.model_env, "").strip()
    # Keep old Fireworks setups that used MODEL_ID working.
    if provider == "fireworks" and not model:
        model = os.getenv("MODEL_ID", "").strip()
    return ProviderConfig(
        provider=provider,
        api_key=os.getenv(spec.key_env, "").strip(),
        model=model or spec.default_model,
        base_url=os.getenv(spec.base_env, "").strip() or spec.default_base_url,
    )


class ProviderSettings:
    """Per-provider profiles stored in QSettings on the local machine."""

    PREFIX = "AI Chat/providers/"

    def __init__(self, store):
        self.store = store

    def selected(self):
        selected = str(self.store.value(self.PREFIX + "selected", "") or "")
        return selected if selected in PROVIDERS else default_provider()

    def select(self, provider):
        if provider not in PROVIDERS:
            raise RuntimeError("Choose a supported AI provider.")
        self.store.setValue(self.PREFIX + "selected", provider)
        self.store.sync()

    def get(self, provider):
        config = environment_config(provider)
        for name in ("api_key", "model", "base_url"):
            value = str(self.store.value(self.PREFIX + provider + "/" + name, "") or "").strip()
            if value:
                setattr(config, name, value)
        return config

    def save(self, config):
        config.validate()
        for name in ("api_key", "model", "base_url"):
            self.store.setValue(self.PREFIX + config.provider + "/" + name, getattr(config, name).strip())
        self.select(config.provider)


def create_client(config):
    config.validate()
    spec = PROVIDERS[config.provider]
    if spec.protocol == "openai":
        from .openai_compatible import CompatibleClient
        return CompatibleClient(config.api_key, config.base_url), config.model
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("Missing AI dependency: pip install anthropic") from None
    return Anthropic(api_key=config.api_key, base_url=config.base_url), config.model


def configured_client(provider=None, model_override="", config=None):
    load_environment()
    config = config or environment_config(provider or default_provider())
    if model_override:
        config = replace(config, model=model_override.strip())
    return create_client(config)
