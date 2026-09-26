"""Provider registry for the multi-provider key manager
(app/core/api_keys_store.py, app/api/api_keys.py) and app/core/llm.py's
key resolution. One place that knows: what fields a given provider's
credentials need, which of those are secret (masked on display, encrypted
at rest), how to cheaply validate a set of credentials without spending
real generation quota, and how to turn stored credentials into the extra
kwargs litellm.completion() needs. Add a provider by adding one entry to
PROVIDERS (plus a branch in validate_credentials/litellm_kwargs if its
credential shape needs a provider-specific call). Nothing else in the app
needs to know its field shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    secret: bool = True
    placeholder: str = ""


# Every modality a provider supports (text, vision, audio, ...) uses this
# one credential set; providers don't split keys by modality, only by
# account. See the /apis page docstring in app/api/main.py.
PROVIDERS: dict[str, list[Field]] = {
    "openai": [Field("api_key", "API key")],
    "anthropic": [Field("api_key", "API key")],
    "gemini": [Field("api_key", "API key")],
    "mistral": [Field("api_key", "API key")],
    "azure_openai": [
        Field("api_key", "API key"),
        Field(
            "api_base",
            "Endpoint URL",
            secret=False,
            placeholder="https://your-resource.openai.azure.com",
        ),
        Field("api_version", "API version", secret=False, placeholder="2024-10-21"),
        Field("deployment", "Deployment name", secret=False),
    ],
    "bedrock": [
        Field("aws_access_key_id", "AWS access key ID"),
        Field("aws_secret_access_key", "AWS secret access key"),
        Field("aws_region_name", "AWS region", secret=False, placeholder="us-east-1"),
    ],
    "ollama": [
        Field("api_base", "Server URL", secret=False, placeholder="http://localhost:11434"),
    ],
}

PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "mistral": "Mistral",
    "azure_openai": "Azure OpenAI",
    "bedrock": "AWS Bedrock",
    "ollama": "Ollama (local)",
}


def fields_for(provider: str) -> list[Field]:
    try:
        return PROVIDERS[provider]
    except KeyError as e:
        raise ValueError(f"unknown provider {provider!r}") from e


def model_prefix_for_provider(provider: str) -> str:
    """Inverse of provider_of_model(): the litellm model-string prefix a
    given provider key actually shows up under, for filtering LLMCall.model
    by provider (app/core/llm.py's per-key budget check)."""
    return "azure" if provider == "azure_openai" else provider


def provider_of_model(model: str) -> str:
    """"openai/gpt-4o-mini" -> "openai". A bare model name with no prefix
    defaults to "openai", litellm's own default too. litellm's model-
    string prefix for Azure is "azure"; ours is "azure_openai" (matches
    the provider key used everywhere else here), so that one prefix is
    remapped.
    """
    if "/" not in model:
        return "openai"
    prefix = model.split("/", 1)[0]
    return "azure_openai" if prefix == "azure" else prefix


def litellm_kwargs(provider: str, credentials: dict) -> dict:
    """Provider-shaped credentials -> the extra kwargs litellm.completion()
    needs beyond model/messages. Passed explicitly per call rather than
    through an env var, so multiple stored keys for the same provider
    never fight over shared global state between calls."""
    if provider == "azure_openai":
        return {
            "api_key": credentials["api_key"],
            "api_base": credentials["api_base"],
            "api_version": credentials["api_version"],
        }
    if provider == "bedrock":
        return {
            "aws_access_key_id": credentials["aws_access_key_id"],
            "aws_secret_access_key": credentials["aws_secret_access_key"],
            "aws_region_name": credentials["aws_region_name"],
        }
    if provider == "ollama":
        return {"api_base": credentials["api_base"]}
    return {"api_key": credentials["api_key"]}


CheckStatus = str  # "valid" | "invalid" | "rate_limited" | "blocked" | "unknown"

# Phrases a provider uses when it is refusing the credential itself
# rather than one request: a suspended, revoked, or not-yet-enabled key.
# Matched against a 403/PermissionDenied message to tell that apart from
# "this key may not use that model", which is a per-model permission and
# says nothing about the key's health.
_BLOCKED_PHRASES = (
    "suspended",
    "revoked",
    "disabled",
    "deactivated",
    "blocked",
    "has been deleted",
    "api key not valid",
    "api_key_invalid",
    "consumer_suspended",
    "account is not active",
    "billing",
    "has not been used in project",
    "api has not been enabled",
    "serviceusage",
)


def is_blocked_detail(detail: str | None) -> bool:
    """Whether a provider's forbidden-class message is about the key being
    shut off (see _BLOCKED_PHRASES). Used by app/core/llm.py to decide
    between the "blocked" status, which is never rechecked on its own, and
    a per-model permission problem, which leaves the key's status alone."""
    low = (detail or "").lower()
    return any(phrase in low for phrase in _BLOCKED_PHRASES)


def validate_credentials(provider: str, credentials: dict) -> tuple[CheckStatus, str]:
    """Cheap check using a list-models or reachability endpoint, never a
    billed completion call. "unknown" means "couldn't check" (network
    error, or Bedrock's no-cheap-call case below); never treat that the
    same as a confirmed-bad "invalid" key. "rate_limited" means the check
    call itself got a 429: the credentials may be fine, just out of quota
    right now, and the returned detail keeps the provider's own words
    because that is where the reset information is (app/core/key_cooldown.py
    reads it). "blocked" means the provider forbade the credential
    outright, which no waiting fixes. Real dispatch (app/core/llm.py) can
    also set any of these, via record_dispatch_outcome().

    A key whose generation quota is spent still answers a list-models call
    with 200, so "valid" here means "this credential is live", not "this
    key has quota left"; only a real dispatch can tell the latter. That is
    what the cooldown in app/core/key_cooldown.py exists for.

    AWS Bedrock has no such lightweight call without a full AWS SigV4
    client (boto3 isn't a dependency here), so its credentials are
    accepted as "unknown" and only proven right/wrong by a real dispatch
    later.
    """
    fields = fields_for(provider)
    missing = [f.label for f in fields if not credentials.get(f.name, "").strip()]
    if missing:
        return "invalid", f"Missing: {', '.join(missing)}"

    if provider == "bedrock":
        return (
            "unknown",
            "Saved. AWS credentials aren't checked here (no AWS SDK dependency); "
            "status updates the first time a real call uses this key.",
        )

    try:
        if provider == "openai":
            r = httpx.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {credentials['api_key']}"},
                timeout=10.0,
            )
        elif provider == "anthropic":
            r = httpx.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": credentials["api_key"],
                    "anthropic-version": "2023-06-01",
                },
                timeout=10.0,
            )
        elif provider == "gemini":
            r = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": credentials["api_key"]},
                timeout=10.0,
            )
        elif provider == "mistral":
            r = httpx.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {credentials['api_key']}"},
                timeout=10.0,
            )
        elif provider == "azure_openai":
            r = httpx.get(
                f"{credentials['api_base'].rstrip('/')}/openai/deployments",
                params={"api-version": credentials["api_version"]},
                headers={"api-key": credentials["api_key"]},
                timeout=10.0,
            )
        elif provider == "ollama":
            r = httpx.get(f"{credentials['api_base'].rstrip('/')}/api/tags", timeout=5.0)
        else:
            return "invalid", f"unknown provider {provider!r}"
    except httpx.RequestError:
        return "unknown", "Couldn't reach the provider to check this. Try again."

    if r.status_code == 200:
        return "valid", "This key is working."
    if r.status_code == 429:
        return (
            "rate_limited",
            "Rate-limited or out of quota right now. The key itself may be fine. "
            f"Provider said: {_body_excerpt(r)}",
        )
    if r.status_code == 403:
        return (
            "blocked",
            "The provider is refusing this key outright (suspended, revoked, or its API not "
            "enabled for this project). Waiting will not fix it. "
            f"Provider said: {_body_excerpt(r)}",
        )
    if r.status_code in (400, 401):
        return "invalid", "This key is invalid. Please check it and try again."
    return "unknown", f"Provider returned an unexpected error (HTTP {r.status_code}). Try again."


def _body_excerpt(response: httpx.Response, limit: int = 300) -> str:
    """The provider's own error body, trimmed to something a page can show
    on one line. Kept verbatim otherwise: for a 429 it is what names the
    quota that was hit and how long to wait."""
    try:
        text = " ".join(response.text.split())
    except Exception:
        return "(no details)"
    if not text:
        return "(no details)"
    return text if len(text) <= limit else text[: limit - 3] + "..."
