"""Read model IDs from a configured provider without making inference calls."""

from urllib.parse import urlsplit

import requests

from .providers import PROVIDERS


def list_models(config, cancelled=lambda: False):
    """Return sorted IDs, following the supported providers' catalog pages.

    Fireworks exposes its catalog through the account Models API, including
    when it is configured as a custom Chat Completions provider. Other custom
    endpoints use the standard /models resource beneath their configured base.
    """
    config.validate(require_model=False)
    base = config.base_url.rstrip("/")
    parsed = urlsplit(base)
    fireworks = parsed.hostname == "api.fireworks.ai"
    featherless = parsed.hostname == "api.featherless.ai"
    native = PROVIDERS[config.provider].protocol == "anthropic" and not fireworks
    headers = {"Accept": "application/json"}
    if native:
        headers.update({"x-api-key": config.api_key, "anthropic-version": "2023-06-01"})
    else:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if fireworks:
        url = f"{parsed.scheme}://{parsed.netloc}/v1/accounts/fireworks/models"
        params = {"pageSize": 200}
    elif native:
        url = base + (
            "/models" if parsed.path.rstrip("/").endswith("/v1") else "/v1/models"
        )
        params = {"limit": 1000}
    else:
        url, params = base + "/models", {}
        if featherless:
            params = {
                "per_page": 1000,
                "page": 1,
                "available_on_current_plan": "true",
                "capabilities": "chat,tool-use",
            }

    ids, seen_pages = set(), set()
    for _ in range(50):
        if cancelled():
            return []
        page_key = tuple(sorted(params.items()))
        if page_key in seen_pages:
            break
        seen_pages.add(page_key)
        try:
            # Keep custom credentials on the configured host, even if an
            # endpoint responds with a redirect to a different catalog.
            with requests.get(
                url,
                headers=headers,
                params=params,
                timeout=(5, 20),
                allow_redirects=False,
            ) as response:
                status = response.status_code
                if status != 200:
                    hint = {
                        401: "Check your API key.",
                        403: "Check your key's model-list permissions.",
                        404: "Check the API base URL; this provider may not support listing models.",
                        429: "Rate limit reached. Try again shortly.",
                    }.get(
                        status,
                        "Check the provider's model-list endpoint or enter a model ID manually.",
                    )
                    raise RuntimeError(f"Could not load models (HTTP {status}). {hint}")
                data = response.json()
        except requests.Timeout:
            raise RuntimeError(
                "Loading models timed out. Try Refresh again or enter a model ID."
            ) from None
        except requests.RequestException:
            raise RuntimeError(
                "Could not connect to load models. Check the base URL and your connection."
            ) from None
        except ValueError:
            raise RuntimeError(
                "The provider did not return a valid model list. You can enter a model ID manually."
            ) from None
        if cancelled():
            return []
        if not isinstance(data, dict):
            raise RuntimeError(
                "The provider returned an unsupported model list. Enter a model ID manually."
            )
        entries = data.get("models" if fireworks else "data", [])
        if not isinstance(entries, list):
            raise RuntimeError(
                "The provider returned an unsupported model list. Enter a model ID manually."
            )
        previous_count = len(ids)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Avoid offering models explicitly unavailable for inference.
            if entry.get("available_on_current_plan") is False:
                continue
            if fireworks and (
                entry.get("supportsServerless") is False
                or entry.get("supportsTools") is False
            ):
                continue
            model_id = entry.get("name" if fireworks else "id")
            if isinstance(model_id, str) and model_id.strip():
                ids.add(model_id.strip())
        if fireworks and data.get("nextPageToken"):
            params["pageToken"] = data["nextPageToken"]
        elif native and data.get("has_more") and data.get("last_id"):
            params["after_id"] = data["last_id"]
        elif featherless and len(entries) == 1000 and len(ids) > previous_count:
            params["page"] += 1
        else:
            break
    return sorted(ids, key=str.casefold)
