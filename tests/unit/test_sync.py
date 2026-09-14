import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import app.core.db as db_module
from app.core.db import Account, Repository, init_db
from app.core.settings import get_settings
from app.ingest.github.cancellation import is_cancelled, request_cancel
from app.ingest.github.sync import (
    _last_authored_commit,
    sync_account,
    sync_account_progress,
    sync_single_repo,
    sync_single_repo_progress,
)


@dataclass
class FakeAuthor:
    login: str


@dataclass
class FakeWeek:
    w: dt.datetime
    c: int


@dataclass
class FakeStatsEntry:
    author: FakeAuthor
    total: int
    weeks: list = field(default_factory=list)


@dataclass
class FakeContentEntry:
    type: str
    name: str
    path: str


@dataclass
class FakeRepo:
    id: int
    name: str
    full_name: str
    html_url: str
    fork: bool
    language: str
    stargazers_count: int
    pushed_at: dt.datetime
    description: str | None = None


class FakeClient:
    """Duck-types GitHubClient without hitting the network."""

    def __init__(
        self,
        repos: list[FakeRepo],
        manifest_files: dict[str, str] | None = None,
        raise_after: int | None = None,
        raise_exc: Exception | None = None,
    ):
        self._repos = repos
        self._manifest_files = manifest_files or {}
        self.stats_calls = 0
        # simulates GitHub cutting us off partway through pagination:
        # yields `raise_after` repos, then raises `raise_exc`
        self._raise_after = raise_after
        self._raise_exc = raise_exc

    def user_repos(self, username: str, include_forks: bool = True):
        for i, repo in enumerate(self._repos):
            if self._raise_after is not None and i >= self._raise_after:
                raise self._raise_exc
            yield repo

    def repo_count_hint(self, username: str) -> int:
        return len(self._repos)

    def get_repo(self, full_name: str):
        for repo in self._repos:
            if repo.full_name == full_name:
                return repo
        raise AssertionError(f"get_repo called unexpectedly for {full_name!r}")

    def readme_text(self, repo):
        return f"# {repo.name}"

    def root_contents(self, repo):
        return [
            FakeContentEntry(type="file", name=name, path=name)
            for name in self._manifest_files
        ]

    def file_text(self, repo, path: str):
        return self._manifest_files.get(path)

    def contributor_stats(self, repo):
        self.stats_calls += 1
        return [
            FakeStatsEntry(
                author=FakeAuthor(login="octocat"),
                total=5,
                weeks=[FakeWeek(w=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), c=5)],
            )
        ]


def _reset_db(tmp_path: Path):
    db_module._engine = None
    db_module._SessionLocal = None
    get_settings.cache_clear()
    import os

    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def test_sync_persists_repos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=3,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    client = FakeClient([repo], manifest_files={"requirements.txt": "requests==2.31.0\n"})

    summary = sync_account("octocat", client=client)

    assert summary.total_repos == 1
    assert summary.fetched == 1
    assert summary.cache_hits == 0
    assert client.stats_calls == 1

    db = db_module.get_db()
    stored = db.query(Repository).filter_by(github_id=1).one()
    assert stored.full_name == "octocat/proj"
    assert stored.commits_authored == 5
    assert stored.manifests_json == {
        "requirements.txt": {"ecosystem": "pip", "dependencies": ["requests"]}
    }
    db.close()


def test_sync_second_run_hits_cache_when_pushed_at_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=3,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    client = FakeClient([repo], manifest_files={"requirements.txt": "requests==2.31.0\n"})

    sync_account("octocat", client=client)
    assert client.stats_calls == 1

    second = sync_account("octocat", client=client)

    assert second.cache_hits == 1
    assert second.fetched == 0
    # cache hit must not refetch readme/manifests/stats
    assert client.stats_calls == 1


def test_sync_captures_description(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=3,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
        description="A tiny thing that does one thing well.",
    )
    client = FakeClient([repo])
    sync_account("octocat", client=client)

    db = db_module.get_db()
    stored = db.query(Repository).filter_by(github_id=1).one()
    assert stored.description == "A tiny thing that does one thing well."
    db.close()


def test_sync_stamps_account_id_when_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=3,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    client = FakeClient([repo])
    sync_account("octocat", client=client, account_id=42)

    db = db_module.get_db()
    stored = db.query(Repository).filter_by(github_id=1).one()
    assert stored.account_id == 42
    db.close()


def test_sync_without_account_id_leaves_it_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=3,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    client = FakeClient([repo])
    sync_account("octocat", client=client)

    db = db_module.get_db()
    stored = db.query(Repository).filter_by(github_id=1).one()
    assert stored.account_id is None
    db.close()


def test_sync_marks_new_repo_pending_extraction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=3,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    client = FakeClient([repo])
    sync_account("octocat", client=client)

    db = db_module.get_db()
    stored = db.query(Repository).filter_by(github_id=1).one()
    assert stored.skill_extraction_status == "pending"
    db.close()


def test_sync_refetch_resets_stale_extraction_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=3,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    client = FakeClient([repo])
    sync_account("octocat", client=client)

    db = db_module.get_db()
    stored = db.query(Repository).filter_by(github_id=1).one()
    stored.skill_extraction_status = "extracted"
    stored.skill_extraction_error = None
    db.commit()
    db.close()

    # content changed -> refetch -> prior extraction is stale
    repo.pushed_at = dt.datetime(2024, 7, 1, tzinfo=dt.UTC)
    sync_account("octocat", client=client)

    db = db_module.get_db()
    stored = db.query(Repository).filter_by(github_id=1).one()
    assert stored.skill_extraction_status == "pending"
    db.close()


def test_sync_cache_hit_leaves_extraction_status_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=3,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    client = FakeClient([repo])
    sync_account("octocat", client=client)

    db = db_module.get_db()
    stored = db.query(Repository).filter_by(github_id=1).one()
    stored.skill_extraction_status = "extracted"
    db.commit()
    db.close()

    # pushed_at unchanged -> cache hit -> extraction status untouched
    sync_account("octocat", client=client)

    db = db_module.get_db()
    stored = db.query(Repository).filter_by(github_id=1).one()
    assert stored.skill_extraction_status == "extracted"
    db.close()


def test_sync_refetches_when_pushed_at_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=3,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    client = FakeClient([repo])
    sync_account("octocat", client=client)

    repo.pushed_at = dt.datetime(2024, 7, 1, tzinfo=dt.UTC)
    second = sync_account("octocat", client=client)

    assert second.fetched == 1
    assert second.cache_hits == 0
    assert client.stats_calls == 2


def test_sync_progress_yields_expected_stage_sequence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repos = [
        FakeRepo(
            id=i,
            name=f"proj{i}",
            full_name=f"octocat/proj{i}",
            html_url=f"https://github.com/octocat/proj{i}",
            fork=False,
            language="Python",
            stargazers_count=0,
            pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
        )
        for i in range(1, 3)
    ]
    client = FakeClient(repos)

    events = list(sync_account_progress("octocat", client=client))

    stages = [e["stage"] for e in events]
    assert stages == ["checking_profile", "listing_repos", "repo_progress", "repo_progress", "done"]
    assert events[1]["total_hint"] == 2
    assert [e["index"] for e in events if e["stage"] == "repo_progress"] == [1, 2]
    assert [e["name"] for e in events if e["stage"] == "repo_progress"] == [
        "octocat/proj1",
        "octocat/proj2",
    ]
    done = events[-1]
    assert done == {"stage": "done", "total_repos": 2, "fetched": 2, "cache_hits": 0}


def test_sync_progress_reports_cache_hits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = FakeRepo(
        id=1,
        name="proj",
        full_name="octocat/proj",
        html_url="https://github.com/octocat/proj",
        fork=False,
        language="Python",
        stargazers_count=0,
        pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    client = FakeClient([repo])
    sync_account("octocat", client=client)  # first sync, primes cache

    events = list(sync_account_progress("octocat", client=client))
    progress_events = [e for e in events if e["stage"] == "repo_progress"]
    assert progress_events[0]["cache_hit"] is True


def _make_repos(n: int) -> list[FakeRepo]:
    return [
        FakeRepo(
            id=i,
            name=f"proj{i}",
            full_name=f"octocat/proj{i}",
            html_url=f"https://github.com/octocat/proj{i}",
            fork=False,
            language="Python",
            stargazers_count=0,
            pushed_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
        )
        for i in range(1, n + 1)
    ]


def test_sync_progress_reports_rate_limited_mid_batch_with_partial_count(tmp_path, monkeypatch):
    from github import RateLimitExceededException

    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    client = FakeClient(
        _make_repos(8),
        raise_after=3,
        raise_exc=RateLimitExceededException(403, "rate limit exceeded", {}),
    )

    events = list(sync_account_progress("octocat", client=client))

    stages = [e["stage"] for e in events]
    assert stages == [
        "checking_profile",
        "listing_repos",
        "repo_progress",
        "repo_progress",
        "repo_progress",
        "rate_limited",
    ]
    assert "done" not in stages  # never fully finished

    rate_limited = events[-1]
    assert rate_limited["completed"] == 3
    assert rate_limited["total_hint"] == 8
    assert "3 of 8" in rate_limited["detail"]

    # the 3 already processed really did land in the DB, nothing lost
    db = db_module.get_db()
    assert db.query(Repository).count() == 3
    db.close()


def test_sync_progress_rate_limited_via_generic_403_githubexception(tmp_path, monkeypatch):
    from github import GithubException

    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    client = FakeClient(
        _make_repos(5), raise_after=2, raise_exc=GithubException(403, "secondary rate limit", {})
    )

    events = list(sync_account_progress("octocat", client=client))

    assert events[-1]["stage"] == "rate_limited"
    assert events[-1]["completed"] == 2


def test_sync_progress_non_rate_limit_error_mid_batch_still_raises(tmp_path, monkeypatch):
    from github import GithubException

    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    client = FakeClient(
        _make_repos(5), raise_after=2, raise_exc=GithubException(500, "server error", {})
    )

    gen = sync_account_progress("octocat", client=client)
    events = []
    raised = None
    try:
        for event in gen:
            events.append(event)
    except GithubException as e:
        raised = e

    assert raised is not None  # a genuine non-rate-limit error still propagates
    assert not any(e["stage"] == "rate_limited" for e in events)
    # the 2 processed before the error still landed
    db = db_module.get_db()
    assert db.query(Repository).count() == 2
    db.close()


def test_sync_progress_stops_when_cancelled_mid_batch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    client = FakeClient(_make_repos(5))

    gen = sync_account_progress("octocat", client=client, run_id="test-run-1")
    events = []
    for event in gen:
        events.append(event)
        if event["stage"] == "repo_progress" and event["index"] == 2:
            request_cancel("test-run-1")

    stages = [e["stage"] for e in events]
    assert stages[-1] == "cancelled"
    assert "done" not in stages
    assert events[-1]["completed"] == 2
    assert events[-1]["total_hint"] == 5

    # flag cleared once acted on; doesn't leak into a future sync of the same id
    assert is_cancelled("test-run-1") is False

    # only the 2 processed before the stop landed, nothing wasted on repo 3+
    db = db_module.get_db()
    assert db.query(Repository).count() == 2
    db.close()


def test_sync_progress_without_run_id_cannot_be_cancelled(tmp_path, monkeypatch):
    """No run_id given -> the (module-level, keyed-by-string) cancellation
    registry is never consulted -> a matching request_cancel call for some
    unrelated id has no effect on this run."""
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    client = FakeClient(_make_repos(3))

    events = list(sync_account_progress("octocat", client=client))  # no run_id

    assert events[-1]["stage"] == "done"
    assert events[-1]["total_repos"] == 3


def test_sync_progress_pending_cancel_flag_cancels_immediately(tmp_path, monkeypatch):
    """A cancel flag already set for run_id before the generator even
    starts iterating takes effect at the very first check. This is what
    makes "click Stop right after clicking Sync" work. Documents the flip
    side too, in the docstring: reusing the same run_id across separate
    sync attempts (instead of a fresh one per attempt) means a flag left
    over from a previous attempt would cancel a brand new one just as
    fast, so callers must use a unique run_id per attempt, not a fixed id.
    """
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    client = FakeClient(_make_repos(3))
    request_cancel("run-x")

    events = list(sync_account_progress("octocat", client=client, run_id="run-x"))

    assert events[-1]["stage"] == "cancelled"
    assert events[-1]["completed"] == 0
    assert is_cancelled("run-x") is False  # cleared once acted on

    db = db_module.get_db()
    assert db.query(Repository).count() == 0
    db.close()


def test_sync_single_repo_progress_cancelled_before_fetch_starts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = _make_repos(1)[0]
    client = FakeClient([repo])
    request_cancel("repo-run-1")

    events = list(
        sync_single_repo_progress(
            repo.full_name, "octocat", client=client, run_id="repo-run-1"
        )
    )

    stages = [e["stage"] for e in events]
    assert stages == ["checking_profile", "cancelled"]
    assert events[-1]["completed"] == 0

    # client.get_repo() was never called (FakeClient would raise if it were)
    # and nothing was written
    db = db_module.get_db()
    assert db.query(Repository).count() == 0
    db.close()
    assert is_cancelled("repo-run-1") is False


def test_last_authored_commit_matches_login_case_insensitively():
    naive_week = dt.datetime(2024, 3, 4)  # no tzinfo: treated as UTC
    epoch_week = int(dt.datetime(2024, 5, 6, tzinfo=dt.UTC).timestamp())
    empty_week = dt.datetime(2024, 9, 2, tzinfo=dt.UTC)
    stats = [
        FakeStatsEntry(author=None, total=99),  # deleted GitHub account
        FakeStatsEntry(author=FakeAuthor(login="someone-else"), total=50),
        FakeStatsEntry(
            author=FakeAuthor(login="OctoCat"),
            total=7,
            weeks=[
                FakeWeek(w=naive_week, c=3),
                FakeWeek(w=epoch_week, c=4),
                FakeWeek(w=empty_week, c=0),  # no commits that week: ignored
            ],
        ),
    ]

    assert _last_authored_commit(stats, "octocat") == (
        7,
        dt.datetime(2024, 5, 6, tzinfo=dt.UTC),
    )


def test_last_authored_commit_naive_week_comes_back_as_utc():
    stats = [
        FakeStatsEntry(
            author=FakeAuthor(login="octocat"),
            total=1,
            weeks=[FakeWeek(w=dt.datetime(2024, 3, 4), c=1)],
        )
    ]

    assert _last_authored_commit(stats, "octocat") == (1, dt.datetime(2024, 3, 4, tzinfo=dt.UTC))


def test_last_authored_commit_without_a_matching_author_is_zero():
    assert _last_authored_commit([], "octocat") == (0, None)
    assert _last_authored_commit(
        [FakeStatsEntry(author=FakeAuthor(login="someone-else"), total=3)], "octocat"
    ) == (0, None)


class _MixedContentsClient(FakeClient):
    """Root listing with a directory, a non-manifest file, a readable
    manifest, and a manifest whose content can't be fetched."""

    def root_contents(self, repo):
        return [
            FakeContentEntry(type="dir", name="package.json", path="package.json"),
            FakeContentEntry(type="file", name="notes.txt", path="notes.txt"),
            FakeContentEntry(type="file", name="requirements.txt", path="requirements.txt"),
            FakeContentEntry(type="file", name="Cargo.toml", path="Cargo.toml"),
        ]

    def file_text(self, repo, path):
        return {"requirements.txt": "flask\n", "notes.txt": "flask\n"}.get(path)


def test_sync_reads_only_fetchable_manifest_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = _make_repos(1)[0]

    sync_account("octocat", client=_MixedContentsClient([repo]))

    db = db_module.get_db()
    stored = db.query(Repository).one()
    assert stored.manifests_json == {
        "requirements.txt": {"ecosystem": "pip", "dependencies": ["flask"]}
    }
    db.close()


def test_sync_naive_pushed_at_still_hits_cache_on_next_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = _make_repos(1)[0]
    repo.pushed_at = dt.datetime(2024, 6, 1, 12, 0)  # naive, as some API paths return
    client = FakeClient([repo])

    sync_account("octocat", client=client)
    second = sync_account("octocat", client=client)

    assert (second.fetched, second.cache_hits) == (0, 1)


def test_sync_refetch_moves_repo_to_the_syncing_account(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    db = db_module.get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    account_id = account.id
    db.close()
    repo = _make_repos(1)[0]
    client = FakeClient([repo])

    sync_account("octocat", client=client)
    repo.pushed_at = repo.pushed_at + dt.timedelta(days=1)
    sync_account("octocat", client=client, account_id=account_id)

    db = db_module.get_db()
    assert db.query(Repository).one().account_id == account_id
    db.close()


class _StatsPendingClient(FakeClient):
    """GitHub still computing contributor stats: returns an empty list."""

    def contributor_stats(self, repo):
        self.stats_calls += 1
        return []


def test_sync_keeps_known_commit_stats_when_github_is_still_computing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = _make_repos(1)[0]
    sync_account("octocat", client=FakeClient([repo]))

    repo.pushed_at = repo.pushed_at + dt.timedelta(days=1)
    sync_account("octocat", client=_StatsPendingClient([repo]))

    db = db_module.get_db()
    stored = db.query(Repository).one()
    assert stored.commits_authored == 5
    assert stored.last_commit_at is not None
    db.close()


def test_sync_single_repo_credits_the_attribution_user_then_hits_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    mine, theirs = _make_repos(2)
    client = FakeClient([mine, theirs])

    first = sync_single_repo(mine.full_name, "octocat", client=client)
    assert (first.total_repos, first.fetched, first.cache_hits) == (1, 1, 0)

    second = sync_single_repo(mine.full_name, "octocat", client=client)
    assert (second.total_repos, second.fetched, second.cache_hits) == (1, 0, 1)

    # contributor stats only list octocat, so another user is credited nothing
    sync_single_repo(theirs.full_name, "ghopper", client=client)

    db = db_module.get_db()
    by_name = {r.full_name: r for r in db.query(Repository).all()}
    assert by_name[mine.full_name].commits_authored == 5
    assert by_name[theirs.full_name].commits_authored == 0
    db.close()


def test_sync_single_repo_progress_full_run_then_cache_hit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_db(tmp_path)
    repo = _make_repos(1)[0]
    client = FakeClient([repo])

    events = list(sync_single_repo_progress(repo.full_name, "octocat", client=client))

    assert [e["stage"] for e in events] == [
        "checking_profile",
        "listing_repos",
        "repo_progress",
        "done",
    ]
    assert events[2] == {
        "stage": "repo_progress",
        "index": 1,
        "total_hint": 1,
        "name": repo.full_name,
        "cache_hit": False,
    }
    assert events[-1]["fetched"] == 1

    again = list(sync_single_repo_progress(repo.full_name, "octocat", client=client))
    assert again[2]["cache_hit"] is True
    assert again[-1]["fetched"] == 0
