"""Tests for app/core/app_settings.py and app/api/app_settings.py: the LLM
models and monthly budget picked on /apis and stored in the database, plus
the rule that `.env` only supplies GITHUB_TOKEN.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import app.core.db as db_module
from app.core import app_settings
from app.core.db import init_db
from app.core.llm_providers import PROVIDERS
from app.core.settings import Settings, get_settings


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


def test_fresh_database_uses_gemini_defaults(tmp_path):
    _reset_db(tmp_path)
    current = app_settings.get_llm_settings()

    assert current.bulk_model == "gemini/gemini-flash-lite-latest"
    assert current.quality_model == "gemini/gemini-flash-latest"
    assert current.monthly_budget_usd == 20.0
    assert current.model_for("bulk") == current.bulk_model
    assert current.model_for("quality") == current.quality_model


def test_update_persists_and_leaves_unset_values_alone(tmp_path):
    _reset_db(tmp_path)
    app_settings.update_llm_settings(bulk_model=" anthropic/claude-haiku-4-5 ")
    app_settings.update_llm_settings(monthly_budget_usd=5)

    current = app_settings.get_llm_settings()
    assert current.bulk_model == "anthropic/claude-haiku-4-5"
    assert current.quality_model == app_settings.DEFAULT_QUALITY_MODEL
    assert current.monthly_budget_usd == 5.0

    app_settings.update_llm_settings(bulk_model="gemini/gemini-2.5-flash")
    assert app_settings.get_llm_settings().bulk_model == "gemini/gemini-2.5-flash"


@pytest.mark.parametrize("model", ["gemini-flash-latest", "gemini/", "/model", "nope/model", " "])
def test_model_without_a_known_provider_is_rejected(tmp_path, model):
    _reset_db(tmp_path)
    with pytest.raises(ValueError):
        app_settings.update_llm_settings(bulk_model=model)


def test_azure_models_use_litellms_azure_prefix(tmp_path):
    _reset_db(tmp_path)
    updated = app_settings.update_llm_settings(quality_model="azure/my-deployment")
    assert updated.quality_model == "azure/my-deployment"


def test_one_invalid_value_writes_nothing(tmp_path):
    _reset_db(tmp_path)
    with pytest.raises(ValueError, match="negative"):
        app_settings.update_llm_settings(bulk_model="openai/gpt-4o-mini", monthly_budget_usd=-1)

    assert app_settings.get_llm_settings().bulk_model == app_settings.DEFAULT_BULK_MODEL


def test_complete_uses_the_picked_model_and_budget(tmp_path, monkeypatch):
    from app.core.llm import BudgetExceededError, complete, user_message

    _reset_db(tmp_path)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 1.0, raising=False)
    app_settings.update_llm_settings(
        quality_model="mistral/mistral-large-latest", monthly_budget_usd=0.5
    )
    calls: list = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    result = complete("quality", [user_message("hi")], _completion_fn=fake_completion)
    assert calls[0]["model"] == "mistral/mistral-large-latest"
    assert result.model == "mistral/mistral-large-latest"

    with pytest.raises(BudgetExceededError, match=r"\$0\.50"):
        complete("quality", [user_message("again")], _completion_fn=fake_completion)


def test_llm_settings_endpoints_round_trip(tmp_path):
    _reset_db(tmp_path)
    client = _client()

    body = client.get("/api/app-settings/llm").json()
    assert body["bulk_model"] == body["default_bulk_model"] == app_settings.DEFAULT_BULK_MODEL
    assert body["default_monthly_budget_usd"] == app_settings.DEFAULT_MONTHLY_BUDGET_USD

    resp = client.put(
        "/api/app-settings/llm",
        json={"quality_model": "openai/gpt-4o", "monthly_budget_usd": 12.5},
    )
    assert resp.status_code == 200
    assert resp.json()["quality_model"] == "openai/gpt-4o"

    body = client.get("/api/app-settings/llm").json()
    assert body["monthly_budget_usd"] == 12.5
    assert body["bulk_model"] == app_settings.DEFAULT_BULK_MODEL


def test_llm_settings_endpoint_rejects_bad_values(tmp_path):
    _reset_db(tmp_path)
    resp = _client().put("/api/app-settings/llm", json={"bulk_model": "no-provider"})

    assert resp.status_code == 422
    assert "<provider>/<model>" in resp.json()["detail"]


def test_model_catalog_covers_every_provider(tmp_path):
    _reset_db(tmp_path)
    body = _client().get("/api/app-settings/llm/models").json()
    by_provider = {entry["provider"]: entry for entry in body}

    assert set(by_provider) == set(PROVIDERS)
    assert by_provider["azure_openai"]["prefix"] == "azure"
    gemini = by_provider["gemini"]["models"]
    assert app_settings.DEFAULT_BULK_MODEL in gemini
    assert app_settings.DEFAULT_QUALITY_MODEL in gemini
    for entry in body:
        assert all(m.startswith(entry["prefix"] + "/") and "*" not in m for m in entry["models"])


def test_dotenv_only_supplies_github_token(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "GITHUB_TOKEN=from-dotenv\n"
        "DATABASE_URL=sqlite:///elsewhere.db\n"
        "QDRANT_URL=http://elsewhere:1\n"
        "LLM_BULK_MODEL=openai/gpt-4o-mini\n"
    )
    monkeypatch.chdir(tmp_path)
    for name in ("GITHUB_TOKEN", "DATABASE_URL", "QDRANT_URL"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()
    assert settings.github_token == "from-dotenv"
    assert settings.database_url == "sqlite:///./data/open_to_work.db"
    assert settings.qdrant_url == "http://localhost:6333"

    monkeypatch.setenv("QDRANT_URL", ":memory:")
    assert Settings().qdrant_url == ":memory:"


@pytest.mark.parametrize(
    "model", ["gemini/AIzaSyExampleExampleExampleExample0", "openai/sk-proj-example", "AIzaSyX"]
)
def test_api_key_pasted_as_a_model_is_rejected(tmp_path, model):
    _reset_db(tmp_path)
    with pytest.raises(ValueError, match="looks like an API key"):
        app_settings.update_llm_settings(bulk_model=model)

    assert app_settings.get_llm_settings().bulk_model == app_settings.DEFAULT_BULK_MODEL


def test_invalid_stored_model_is_never_dispatched(tmp_path):
    from app.core.db import AppSetting, get_db

    _reset_db(tmp_path)
    db = get_db()
    try:
        db.add(AppSetting(key="llm_bulk_model", value="gemini/AIzaSyExampleExampleExample"))
        db.commit()
    finally:
        db.close()

    assert app_settings.get_llm_settings().bulk_model == app_settings.DEFAULT_BULK_MODEL


def test_a_priced_model_reports_no_budget_warning(tmp_path):
    _reset_db(tmp_path)

    assert app_settings.is_model_priced("gemini/gemini-flash-lite-latest") is True
    assert app_settings.unpriced_models_in_use() == []


def test_a_model_litellm_cannot_price_is_saved_but_flagged(tmp_path):
    """The budget is a sum over recorded cost, and an unpriceable model
    records 0.00 per call, so every cap stops working. Saving it is still
    right: a new release or a local model is a real choice, and
    suggested_models() promises any provider-prefixed string can be picked.
    """
    _reset_db(tmp_path)
    unpriceable = "gemini/gemini-4-ultra-preview-2027"

    assert app_settings.is_model_priced(unpriceable) is False

    updated = app_settings.update_llm_settings(bulk_model=unpriceable)
    assert updated.bulk_model == unpriceable  # accepted, not rejected
    assert app_settings.unpriced_models_in_use() == [unpriceable]


def test_the_settings_endpoint_warns_about_an_unpriceable_model(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    unpriceable = "openai/my-finetune-2027"

    healthy = client.get("/api/app-settings/llm").json()
    assert healthy["unpriced_models"] == []
    assert healthy["budget_warning"] is None

    response = client.put("/api/app-settings/llm", json={"bulk_model": unpriceable})
    assert response.status_code == 200
    body = response.json()
    assert body["bulk_model"] == unpriceable
    assert body["unpriced_models"] == [unpriceable]
    assert "$0.00" in body["budget_warning"]
    assert unpriceable in body["budget_warning"]

    # and it stays flagged on a later read, not only on the save that set it
    assert client.get("/api/app-settings/llm").json()["unpriced_models"] == [unpriceable]


def test_both_tiers_unpriceable_are_listed_once_each(tmp_path):
    _reset_db(tmp_path)
    app_settings.update_llm_settings(
        bulk_model="openai/my-finetune-2027", quality_model="openai/my-finetune-2027"
    )

    assert app_settings.unpriced_models_in_use() == ["openai/my-finetune-2027"]
