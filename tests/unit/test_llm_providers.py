"""Provider registry (app/core/llm_providers.py): field shapes, model-string
parsing, litellm kwargs, and credential validation. The HTTP layer is faked,
so no request ever reaches a real provider."""

import httpx
import pytest

from app.core import llm_providers as lp

_CREDS = {
    "openai": {"api_key": "k-openai"},
    "anthropic": {"api_key": "k-anthropic"},
    "gemini": {"api_key": "k-gemini"},
    "mistral": {"api_key": "k-mistral"},
    "azure_openai": {
        "api_key": "k-azure",
        "api_base": "https://res.openai.azure.com/",
        "api_version": "2024-10-21",
        "deployment": "gpt4o",
    },
    "bedrock": {
        "aws_access_key_id": "AKIA",
        "aws_secret_access_key": "shh",
        "aws_region_name": "us-east-1",
    },
    "ollama": {"api_base": "http://localhost:11434/"},
}


class _FakeGet:
    """Stands in for httpx.get: records each call, answers with one status."""

    def __init__(self, status: int = 200):
        self.status = status
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return httpx.Response(self.status)


def test_credential_fixture_covers_every_registered_provider():
    assert set(_CREDS) == set(lp.PROVIDERS)


def test_every_provider_has_a_label_and_unique_field_names():
    assert set(lp.PROVIDERS) == set(lp.PROVIDER_LABELS)
    for provider, fields in lp.PROVIDERS.items():
        assert fields, provider
        assert len({f.name for f in fields}) == len(fields), provider


def test_secrets_are_marked_secret_and_endpoints_are_not():
    for fields in lp.PROVIDERS.values():
        for f in fields:
            if f.name in ("api_key", "aws_access_key_id", "aws_secret_access_key"):
                assert f.secret, f.name
            if f.name in ("api_base", "api_version", "deployment", "aws_region_name"):
                assert not f.secret, f.name


def test_fields_for_known_and_unknown_provider():
    assert [f.name for f in lp.fields_for("openai")] == ["api_key"]
    with pytest.raises(ValueError, match="unknown provider"):
        lp.fields_for("not-a-provider")


@pytest.mark.parametrize(
    "model, provider",
    [
        ("openai/gpt-4o-mini", "openai"),
        ("gpt-4o-mini", "openai"),
        ("anthropic/some-model", "anthropic"),
        ("gemini/gemini-2.0-flash", "gemini"),
        ("azure/my-deployment", "azure_openai"),
        ("bedrock/amazon.nova-lite", "bedrock"),
        ("ollama/llama3", "ollama"),
    ],
)
def test_provider_of_model(model, provider):
    assert lp.provider_of_model(model) == provider


@pytest.mark.parametrize("provider", sorted(lp.PROVIDERS))
def test_model_prefix_round_trips_through_provider_of_model(provider):
    prefix = lp.model_prefix_for_provider(provider)
    assert lp.provider_of_model(f"{prefix}/any-model") == provider


def test_litellm_kwargs_per_credential_shape():
    assert lp.litellm_kwargs("openai", _CREDS["openai"]) == {"api_key": "k-openai"}
    assert lp.litellm_kwargs("azure_openai", _CREDS["azure_openai"]) == {
        "api_key": "k-azure",
        "api_base": "https://res.openai.azure.com/",
        "api_version": "2024-10-21",
    }
    assert lp.litellm_kwargs("bedrock", _CREDS["bedrock"]) == _CREDS["bedrock"]
    assert lp.litellm_kwargs("ollama", _CREDS["ollama"]) == {"api_base": "http://localhost:11434/"}


@pytest.mark.parametrize(
    "provider, url",
    [
        ("openai", "https://api.openai.com/v1/models"),
        ("anthropic", "https://api.anthropic.com/v1/models"),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/models"),
        ("mistral", "https://api.mistral.ai/v1/models"),
        ("azure_openai", "https://res.openai.azure.com/openai/deployments"),
        ("ollama", "http://localhost:11434/api/tags"),
    ],
)
def test_validate_hits_a_free_list_endpoint(monkeypatch, provider, url):
    fake = _FakeGet(200)
    monkeypatch.setattr(lp.httpx, "get", fake)

    status, detail = lp.validate_credentials(provider, _CREDS[provider])

    assert (status, detail) == ("valid", "This key is working.")
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == url
    assert "timeout" in fake.calls[0][1]


def test_validate_sends_each_provider_key_in_its_expected_place(monkeypatch):
    fake = _FakeGet(200)
    monkeypatch.setattr(lp.httpx, "get", fake)
    for provider in ("openai", "anthropic", "gemini", "mistral", "azure_openai"):
        lp.validate_credentials(provider, _CREDS[provider])

    openai, anthropic, gemini, mistral, azure = (kw for _, kw in fake.calls)
    assert openai["headers"]["Authorization"] == "Bearer k-openai"
    assert anthropic["headers"]["x-api-key"] == "k-anthropic"
    assert "anthropic-version" in anthropic["headers"]
    assert gemini["params"]["key"] == "k-gemini"
    assert mistral["headers"]["Authorization"] == "Bearer k-mistral"
    assert azure["headers"]["api-key"] == "k-azure"
    assert azure["params"]["api-version"] == "2024-10-21"


@pytest.mark.parametrize(
    "http_status, expected",
    [
        (200, "valid"),
        (400, "invalid"),
        (401, "invalid"),
        (403, "invalid"),
        (429, "rate_limited"),
        (404, "unknown"),
        (500, "unknown"),
        (503, "unknown"),
    ],
)
def test_validate_maps_http_status_to_key_status(monkeypatch, http_status, expected):
    monkeypatch.setattr(lp.httpx, "get", _FakeGet(http_status))

    status, _ = lp.validate_credentials("openai", _CREDS["openai"])

    assert status == expected


def test_validate_network_error_is_unknown_not_invalid(monkeypatch):
    def unreachable(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(lp.httpx, "get", unreachable)

    status, detail = lp.validate_credentials("ollama", _CREDS["ollama"])

    assert status == "unknown"
    assert "reach" in detail


def test_validate_missing_or_blank_fields_is_invalid_without_a_request(monkeypatch):
    fake = _FakeGet(200)
    monkeypatch.setattr(lp.httpx, "get", fake)

    status, detail = lp.validate_credentials(
        "azure_openai", {"api_key": "k", "api_base": "   ", "api_version": ""}
    )

    assert status == "invalid"
    assert "Endpoint URL" in detail
    assert "API version" in detail
    assert "Deployment name" in detail
    assert "API key" not in detail
    assert fake.calls == []


def test_validate_bedrock_is_accepted_as_unknown_without_a_request(monkeypatch):
    fake = _FakeGet(200)
    monkeypatch.setattr(lp.httpx, "get", fake)

    status, _ = lp.validate_credentials("bedrock", _CREDS["bedrock"])

    assert status == "unknown"
    assert fake.calls == []


def test_validate_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        lp.validate_credentials("not-a-provider", {"api_key": "k"})
