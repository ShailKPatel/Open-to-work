"""app/api/demo.py and the demo side of app/api/security.py, through the
real app. Mostly attacks: what one signed-in person must not reach."""

import io
import re
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from itsdangerous import URLSafeTimedSerializer

import app.api.demo as demo
from app.api.main import app as fastapi_app
from app.api.security import refusal
from app.core.db import (
    Account,
    ContactEmail,
    Experience,
    ExperiencePoint,
    JobPosting,
    LLMCall,
    Resume,
    SyncSource,
    get_db,
    init_db,
)
from app.core.llm import BudgetExceededError, _enforce_demo_limits
from app.core.settings import Settings, get_settings
from app.core.usage import BUILD_STARTED, RESUME_BUILT, count_today, record
from app.resume_build import background

HOST = "demo.example.com"
BASE = f"https://{HOST}"
SECRET = "s" * 40
ADMIN_GITHUB_ID = 999


@pytest.fixture(autouse=True)
def _demo(tmp_path, monkeypatch):
    import app.core.db as db_module

    db_module.reset_engine()
    for name, value in {
        "DATABASE_URL": f"sqlite:///{tmp_path}/t.db",
        "QDRANT_URL": ":memory:",
        "DEMO_MODE": "true",
        "DEMO_PUBLIC_HOST": HOST,
        "DEMO_GITHUB_CLIENT_ID": "id",
        "DEMO_GITHUB_CLIENT_SECRET": "secret",
        "DEMO_SESSION_SECRET": SECRET,
        "DEMO_ADMIN_GITHUB_ID": str(ADMIN_GITHUB_ID),
        "RESUME_STORAGE_DIR": str(tmp_path / "resumes"),
    }.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    init_db()
    yield
    db_module.reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def world():
    """Two people, each owning one of everything."""
    db = get_db()
    ids = {}
    for who in ("me", "them"):
        account = Account(first_name=who, last_name="x", github_username=who)
        db.add(account)
        db.commit()
        rows = {
            "resume": Resume(account_id=account.id, filename="r.pdf", mime_type="application/pdf"),
            "posting": JobPosting(
                account_id=account.id, source="pasted", external_id=who, company="C", title="T",
                raw_text_quarantined="secret posting of " + who, content_hash=who,
            ),
            "source": SyncSource(
                account_id=account.id, raw_input=who, kind="user", github_username=who
            ),
        }
        db.add_all(rows.values())
        db.commit()
        ids[who] = {"account": account.id, **{k: v.id for k, v in rows.items()}}
    db.close()
    return ids


def _client(account_id=None, admin=False):
    client = TestClient(fastapi_app, base_url=BASE, follow_redirects=False)
    if account_id is not None:
        token = URLSafeTimedSerializer(SECRET, salt="otw-session").dumps(
            {"account_id": account_id, "admin": admin}
        )
        client.cookies.set("otw_session", token, domain=HOST)
    return client


def _plain(path):
    """A route path as the schema spells it, without converters."""
    return path.replace(":int", "")


def _operations():
    """Every operation the app serves. From the schema because included
    routers are not all flattened into app.routes."""
    return {
        (method.upper(), path)
        for path, item in fastapi_app.openapi()["paths"].items()
        for method in item
    }


ALLOWED = {(method, _plain(path)) for method, path in demo._ALLOWED}


def test_every_allowed_route_exists():
    assert not ALLOWED - _operations()


def test_every_route_not_listed_is_closed(world):
    client = _client(world["me"]["account"])
    checked = 0
    for method, path in sorted(_operations() - ALLOWED):
        if path in demo._PUBLIC:
            continue
        response = client.request(method, re.sub(r"\{[^}]+\}", "1", path))
        assert response.status_code == 404, (method, path, response.status_code)
        checked += 1
    assert checked > 25


@pytest.mark.parametrize(
    "path",
    [
        "/api/api-keys",
        "/api/monitor/llm/calls",
        "/api/resume-build/generate",
        "/apis",
        "/settings",
    ],
)
def test_sensitive_routes_are_closed_by_name(world, path):
    client = _client(world["me"]["account"])
    response = client.get(path) if "generate" not in path else client.post(path, json={})
    assert response.status_code == 404


def test_signed_out_api_is_401_and_pages_go_to_sign_in():
    client = _client()
    assert client.get("/api/resume", params={"account_id": 1}).status_code == 401
    page = client.get("/home")
    assert (page.status_code, page.headers["location"]) == (303, "/welcome")


def test_signed_out_write_is_refused_before_the_body_is_read():
    response = _client().post("/api/resume", files={"file": ("a.pdf", io.BytesIO(b"x"))})
    assert response.status_code == 401


def test_forged_cookie_is_not_a_session(world):
    client = TestClient(fastapi_app, base_url=BASE)
    forged = URLSafeTimedSerializer("wrong" * 10, salt="otw-session").dumps(
        {"account_id": world["me"]["account"], "admin": True}
    )
    client.cookies.set("otw_session", forged, domain=HOST)
    assert client.get("/api/api-keys").status_code == 401


def test_session_for_a_deleted_account_is_signed_out():
    assert _client(424242).get("/api/resume", params={"account_id": 424242}).status_code == 401


def test_other_accounts_id_in_query_is_refused(world):
    me, them = world["me"], world["them"]
    client = _client(me["account"])
    assert client.get("/api/resume", params={"account_id": me["account"]}).status_code == 200
    assert client.get("/api/resume", params={"account_id": them["account"]}).status_code == 404


def test_repeating_a_query_parameter_does_not_slip_past(world):
    me, them = world["me"], world["them"]
    url = f"/api/resume?account_id={me['account']}&account_id={them['account']}"
    assert _client(me["account"]).get(url).status_code == 404
    url = f"/api/resume?account_id={them['account']}&account_id={me['account']}"
    assert _client(me["account"]).get(url).status_code == 404


@pytest.mark.parametrize(
    "template",
    [
        "/api/resume/{resume}/file",
        "/api/resume/{resume}/profile",
        "/api/job-postings/{posting}",
        "/api/sources/{source}/sync/stream",
    ],
)
def test_other_accounts_rows_look_missing(world, template):
    client = _client(world["me"]["account"])
    assert client.get(template.format(**world["them"])).status_code == 404


def test_own_posting_is_readable_and_theirs_is_not(world):
    client = _client(world["me"]["account"])
    mine = client.get(f"/api/job-postings/{world['me']['posting']}")
    assert mine.status_code == 200 and "of me" in mine.text


def test_other_accounts_id_in_json_body_is_refused(world):
    me, them = world["me"], world["them"]
    client = _client(me["account"])
    body = {"account_id": them["account"], "job_posting_id": them["posting"]}
    assert client.post("/api/resume-build/start", json=body).status_code == 404
    body = {"account_id": me["account"], "job_posting_id": them["posting"]}
    assert client.post("/api/resume-build/start", json=body).status_code == 404


def test_other_accounts_id_in_upload_form_is_refused_and_own_reaches_handler(world, monkeypatch):
    me, them = world["me"], world["them"]
    client = _client(me["account"])
    file = {"file": ("a.pdf", io.BytesIO(b"%PDF"), "application/pdf")}
    refused = client.post("/api/resume", data={"account_id": them["account"]}, files=file)
    assert refused.status_code == 404

    def _reached(*a, **k):
        raise demo.HTTPException(status_code=418)

    monkeypatch.setattr("app.api.resume.ingest_resume", _reached)
    file = {"file": ("a.pdf", io.BytesIO(b"%PDF"), "application/pdf")}
    reached = client.post("/api/resume", data={"account_id": me["account"]}, files=file)
    assert reached.status_code == 418


def test_other_accounts_build_is_hidden(world):
    build = background.Build(
        id="theirs", account_id=world["them"]["account"], job_posting_id=1, label="l"
    )
    background._builds["theirs"] = build
    try:
        client = _client(world["me"]["account"])
        assert client.get("/api/resume-build/builds/theirs").status_code == 404
        decision = client.post(
            "/api/resume-build/builds/theirs/decision", json={"action": "approve"}
        )
        assert decision.status_code == 404
        assert _client(world["them"]["account"]).get(
            "/api/resume-build/builds/theirs"
        ).status_code == 200
    finally:
        background._builds.pop("theirs")


def test_account_list_shows_only_your_own(world):
    me = world["me"]["account"]
    assert [a["id"] for a in _client(me).get("/accounts").json()] == [me]


def test_admin_passes_the_gate_and_sees_everyone(world):
    client = _client(world["me"]["account"], admin=True)
    assert client.get("/api/api-keys").status_code == 200
    assert len(client.get("/accounts").json()) == 2


def test_root_goes_home(world):
    assert _client(world["me"]["account"]).get("/").headers["location"] == "/home"


def test_page_identifies_the_session_and_hides_owner_links(world):
    html = _client(world["me"]["account"]).get("/home").text
    assert f"localStorage.setItem('otw_account_id', {world['me']['account']})" in html
    assert "/auth/logout" in html
    assert 'href="/settings"' not in html


def test_every_id_in_an_allowed_path_has_an_owner_rule():
    known = {*demo._OWNERS, "account_id", "build_id", "index"}
    known |= {child[0] for child in demo._CHILDREN.values()}
    for _, path in demo._ALLOWED:
        for param in re.findall(r"\{(\w+)", path):
            assert param in known, path


def test_every_child_path_is_really_allowed():
    allowed = {path for _, path in demo._ALLOWED}
    assert set(demo._CHILDREN) <= allowed


def _experience_with_point(account_id):
    db = get_db()
    experience = Experience(account_id=account_id, title="T", company="C")
    db.add(experience)
    db.commit()
    point = ExperiencePoint(experience_id=experience.id, text="did a thing")
    db.add(point)
    db.commit()
    ids = experience.id, point.id
    db.close()
    return ids


def test_child_id_from_someone_elses_parent_is_refused(world):
    me, them = world["me"]["account"], world["them"]["account"]
    my_role, my_point = _experience_with_point(me)
    _, their_point = _experience_with_point(them)
    client = _client(me)

    stolen = client.patch(
        f"/api/experience/{my_role}/points/{their_point}", json={"text": "hijacked"}
    )
    assert stolen.status_code == 404
    db = get_db()
    assert db.get(ExperiencePoint, their_point).text == "did a thing"
    db.close()

    own = client.patch(f"/api/experience/{my_role}/points/{my_point}", json={"text": "better"})
    assert own.status_code == 200


def test_other_accounts_contact_row_is_refused(world):
    me, them = world["me"]["account"], world["them"]["account"]
    db = get_db()
    email = ContactEmail(account_id=them, email="them@example.com", is_primary=True)
    db.add(email)
    db.commit()
    email_id = email.id
    db.close()
    client = _client(me)
    assert client.delete(f"/api/accounts/{me}/emails/{email_id}").status_code == 404
    assert client.delete(f"/api/accounts/{them}/emails/{email_id}").status_code == 404


def test_creating_for_someone_else_is_refused(world):
    me, them = world["me"]["account"], world["them"]["account"]
    client = _client(me)
    body = {"account_id": them, "title": "T", "company": "C"}
    assert client.post("/api/experience", json=body).status_code == 404
    assert client.post("/api/skills", json={"account_id": them, "name": "Go"}).status_code == 404
    assert client.post(f"/api/sources/sync-all?account_id={them}").status_code == 404


def test_write_limit(world, monkeypatch):
    me = world["me"]["account"]
    monkeypatch.setenv("DEMO_DAILY_WRITES", "1")
    get_settings.cache_clear()
    client = _client(me)
    assert client.post("/api/skills", json={"account_id": me, "name": "Go"}).status_code == 200
    assert client.post("/api/skills", json={"account_id": me, "name": "C"}).status_code == 429


def test_intake_limit(world, monkeypatch):
    me = world["me"]
    monkeypatch.setenv("DEMO_DAILY_INTAKE", "0")
    get_settings.cache_clear()
    response = _client(me["account"]).post(
        f"/api/job-postings/{me['posting']}/reprocess"
    )
    assert response.status_code == 429


def test_source_limit(world, monkeypatch):
    me = world["me"]["account"]
    monkeypatch.setenv("DEMO_MAX_SOURCES", "1")
    get_settings.cache_clear()
    body = {"account_id": me, "raw_input": "octocat"}
    assert _client(me).post("/api/sources", json=body).status_code == 429


def test_global_build_limit(world, monkeypatch):
    me = world["me"]
    monkeypatch.setenv("DEMO_MAX_RUNNING_BUILDS", "1")
    get_settings.cache_clear()
    them = world["them"]["account"]
    build = background.Build(id="x", account_id=them, job_posting_id=1, label="l")
    background._builds["x"] = build
    try:
        body = {"account_id": me["account"], "job_posting_id": me["posting"]}
        response = _client(me["account"]).post("/api/resume-build/start", json=body)
        assert response.status_code == 429 and "busy" in response.text
    finally:
        background._builds.pop("x")


def test_oversized_body_is_refused_before_reading(world):
    client = _client(world["me"]["account"])
    response = client.post(
        "/api/resume", headers={"content-length": str(13 * 1024 * 1024)}, content=b"x"
    )
    assert response.status_code == 413


def test_welcome_tells_people_what_happens_to_their_data():
    html = _client().get("/welcome").text
    assert "stored on this server" in html and "/auth/login" in html
    assert "LangSmith" not in html


def test_welcome_discloses_tracing_when_on(monkeypatch):
    monkeypatch.setenv("TRACING_ENABLED", "true")
    get_settings.cache_clear()
    assert "LangSmith" in _client().get("/welcome").text


def test_welcome_offers_deletion_once_signed_in(world):
    html = _client(world["me"]["account"]).get("/welcome").text
    assert "Delete my account" in html and "/auth/login" not in html


def test_deleting_an_account_removes_copies_kept_elsewhere(world):
    me = world["me"]["account"]
    db = get_db()
    db.add(
        LLMCall(
            tier="q", model="m", prompt_hash="h", account_id=me,
            response_json={"content": "resume text"},
        )
    )
    db.commit()
    db.close()
    background._builds["gone"] = background.Build(
        id="gone", account_id=me, job_posting_id=1, label="l"
    )

    assert _client(me).delete(f"/accounts/{me}").status_code == 200

    assert "gone" not in background._builds
    db = get_db()
    row = db.query(LLMCall).filter(LLMCall.prompt_hash == "h").one()
    assert row.response_json is None
    db.close()


def test_explanation_describes_the_hosted_app():
    html = _client().get("/explanation").text
    assert "on one small cloud server" in html and "localhost:8000" not in html


def test_home_does_not_ask_visitors_for_an_api_key(world):
    assert "this.llmConnected = true;" in _client(world["me"]["account"]).get("/home").text


def test_docs_are_closed():
    client = _client()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_health_answers_a_probe_by_ip_but_other_hosts_are_refused():
    client = TestClient(fastapi_app, base_url="http://10.0.0.5:8000")
    assert client.get("/health").status_code == 200
    assert client.get("/explanation").status_code == 400


def test_public_pages_and_hsts():
    response = _client().get("/explanation")
    assert response.status_code == 200
    assert "max-age" in response.headers["strict-transport-security"]


def test_origin_is_https_even_if_the_proxy_speaks_http():
    headers = {"host": HOST, "origin": BASE}
    assert refusal("POST", headers, "http", HOST) is None
    headers = {"host": HOST, "origin": "https://evil.example"}
    assert refusal("POST", headers, "http", HOST)[0] == 403
    assert refusal("GET", {"host": "localhost"}, "http", HOST)[0] == 400


def test_build_limit_per_day(world, monkeypatch):
    me = world["me"]
    monkeypatch.setenv("DEMO_DAILY_RESUME_BUILDS", "1")
    get_settings.cache_clear()
    record(me["account"], BUILD_STARTED)
    body = {"account_id": me["account"], "job_posting_id": me["posting"]}
    response = _client(me["account"]).post("/api/resume-build/start", json=body)
    assert response.status_code == 429
    assert count_today(me["account"], BUILD_STARTED) == 1


def test_one_running_build_at_a_time(world):
    me = world["me"]
    build = background.Build(id="busy", account_id=me["account"], job_posting_id=1, label="l")
    background._builds["busy"] = build
    try:
        body = {"account_id": me["account"], "job_posting_id": me["posting"]}
        assert _client(me["account"]).post("/api/resume-build/start", json=body).status_code == 429
    finally:
        background._builds.pop("busy")


def _spend(account_id, cost, n=1):
    db = get_db()
    for i in range(n):
        db.add(
            LLMCall(
                tier="bulk", model="m", prompt_hash=str(i), cost_usd=cost, account_id=account_id
            )
        )
    db.commit()
    db.close()


def test_daily_cost_cap_stops_model_calls(world):
    me = world["me"]["account"]
    _enforce_demo_limits(me)
    _spend(me, 0.6)
    with pytest.raises(BudgetExceededError, match="demo limit"):
        _enforce_demo_limits(me)
    _enforce_demo_limits(world["them"]["account"])


def test_daily_call_cap_catches_a_model_that_reports_no_cost(world, monkeypatch):
    me = world["me"]["account"]
    monkeypatch.setenv("DEMO_DAILY_LLM_CALLS", "3")
    get_settings.cache_clear()
    _spend(me, 0.0, n=3)
    with pytest.raises(BudgetExceededError):
        _enforce_demo_limits(me)


def test_limits_are_off_locally(world, monkeypatch):
    me = world["me"]["account"]
    _spend(me, 99.0)
    monkeypatch.setenv("DEMO_MODE", "false")
    get_settings.cache_clear()
    _enforce_demo_limits(me)


def test_login_sets_state_and_asks_for_no_scope():
    response = _client().get("/auth/login")
    assert response.status_code == 303
    assert response.headers["location"].endswith("&scope=")
    assert "otw_oauth_state" in response.headers["set-cookie"]


def _callback(monkeypatch, profile):
    monkeypatch.setattr(demo, "_github_profile", MagicMock(return_value=profile))
    client = _client()
    client.cookies.set("otw_oauth_state", "abc", domain=HOST, path="/auth")
    return client.get("/auth/callback", params={"code": "c", "state": "abc"})


def test_callback_needs_the_matching_state(monkeypatch):
    monkeypatch.setattr(demo, "_github_profile", MagicMock(side_effect=AssertionError))
    client = _client()
    assert client.get("/auth/callback", params={"code": "c", "state": "x"}).status_code == 400
    client.cookies.set("otw_oauth_state", "abc", domain=HOST, path="/auth")
    assert client.get("/auth/callback", params={"code": "c", "state": "x"}).status_code == 400


def test_callback_creates_an_account_and_a_renamed_login_keeps_it(monkeypatch):
    first = _callback(monkeypatch, {"id": 5, "login": "ada", "name": "Ada Lovelace"})
    assert (first.status_code, first.headers["location"]) == (303, "/home")
    assert "otw_session" in first.headers["set-cookie"]
    assert "secure" in first.headers["set-cookie"].lower()
    _callback(monkeypatch, {"id": 5, "login": "ada2", "name": None})
    # Someone else now holding the old login is a different person.
    _callback(monkeypatch, {"id": 6, "login": "ada", "name": None})

    db = get_db()
    rows = db.query(Account).order_by(Account.id).all()
    assert [(a.first_name, a.last_name, a.github_user_id) for a in rows] == [
        ("Ada", "Lovelace", 5),
        ("ada", "", 6),
    ]
    db.close()


def test_admin_flag_comes_from_the_github_id(monkeypatch):
    response = _callback(monkeypatch, {"id": ADMIN_GITHUB_ID, "login": "owner", "name": None})
    assert demo.session_account(response.cookies["otw_session"].strip('"'))["admin"] is True


def test_demo_full_refuses_new_people_but_not_the_owner(monkeypatch):
    monkeypatch.setenv("DEMO_MAX_ACCOUNTS", "0")
    get_settings.cache_clear()
    assert _callback(monkeypatch, {"id": 7, "login": "new", "name": None}).status_code == 403
    owner = _callback(monkeypatch, {"id": ADMIN_GITHUB_ID, "login": "o", "name": None})
    assert owner.status_code == 303


def test_logout_clears_the_session(world):
    response = _client(world["me"]["account"]).post("/auth/logout")
    assert response.status_code == 303
    assert 'otw_session=""' in response.headers["set-cookie"]


def test_auth_routes_do_not_exist_locally(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    get_settings.cache_clear()
    client = TestClient(fastapi_app, base_url="http://localhost:8000")
    assert client.get("/auth/login").status_code == 404


@pytest.mark.parametrize(
    "missing", ["demo_public_host", "demo_github_client_id", "demo_session_secret"]
)
def test_incomplete_demo_settings_refuse_to_start(missing):
    good = {
        "demo_mode": True,
        "demo_public_host": HOST,
        "demo_github_client_id": "i",
        "demo_github_client_secret": "s",
        "demo_session_secret": SECRET,
    }
    demo.require_valid_demo_settings(Settings(**good))
    with pytest.raises(RuntimeError, match="not set"):
        demo.require_valid_demo_settings(Settings(**{**good, missing: ""}))


def test_usage_report(world):
    from scripts.usage_report import report

    me = world["me"]["account"]
    record(me, RESUME_BUILT, {"rejected": 3, "saved": 2, "dropped": 1})
    record(me, RESUME_BUILT, {"rejected": 1, "saved": 1, "dropped": 0})
    db = get_db()
    db.add(LLMCall(tier="q", model="m", prompt_hash="p", cost_usd=0.1, purpose="resume_build"))
    db.add(LLMCall(tier="q", model="m", prompt_hash="q", cost_usd=5.0, purpose="repo_facts"))
    db.commit()
    db.close()

    result = report()

    assert result["resumes_built"] == 2 and result["users_with_a_resume"] == 1
    assert (result["bullets_rejected"], result["bullets_saved_by_revision"]) == (4, 3)
    assert result["avg_cost_per_resume_usd"] == 0.05
