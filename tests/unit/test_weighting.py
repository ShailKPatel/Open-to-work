import datetime as dt

from app.profile.weighting import compute_weight

NOW = dt.datetime(2026, 8, 24, tzinfo=dt.UTC)


def test_fresh_active_own_repo_scores_high():
    w = compute_weight(
        "declared_dependency",
        1.0,
        is_fork=False,
        last_commit_at=NOW - dt.timedelta(days=10),
        commits_authored=100,
        now=NOW,
    )
    assert w > 0.4  # base 0.5 * full recency * saturated volume * non-fork


def test_fork_is_heavily_discounted():
    kwargs = dict(
        confidence=1.0,
        last_commit_at=NOW - dt.timedelta(days=10),
        commits_authored=100,
        now=NOW,
    )
    owned = compute_weight("declared_dependency", is_fork=False, **kwargs)
    forked = compute_weight("declared_dependency", is_fork=True, **kwargs)
    assert forked < owned * 0.4


def test_stale_repo_scores_lower_than_recent():
    kwargs = dict(
        evidence_type="declared_dependency",
        confidence=1.0,
        is_fork=False,
        commits_authored=20,
        now=NOW,
    )
    recent = compute_weight(last_commit_at=NOW - dt.timedelta(days=30), **kwargs)
    stale = compute_weight(last_commit_at=NOW - dt.timedelta(days=900), **kwargs)
    assert stale < recent


def test_unknown_last_commit_gets_moderate_discount_not_zero():
    w = compute_weight(
        "declared_dependency",
        1.0,
        is_fork=False,
        last_commit_at=None,
        commits_authored=10,
        now=NOW,
    )
    assert 0.0 < w < 1.0


def test_more_commits_scores_higher_than_few():
    kwargs = dict(
        evidence_type="declared_dependency",
        confidence=1.0,
        is_fork=False,
        last_commit_at=NOW - dt.timedelta(days=10),
        now=NOW,
    )
    few = compute_weight(commits_authored=1, **kwargs)
    many = compute_weight(commits_authored=100, **kwargs)
    assert many > few


def test_weight_always_clamped_to_unit_interval():
    w = compute_weight(
        "readme_described",
        1.0,
        is_fork=False,
        last_commit_at=NOW,
        commits_authored=10_000,
        now=NOW,
    )
    assert 0.0 <= w <= 1.0


def test_low_llm_confidence_lowers_readme_described_weight():
    kwargs = dict(
        evidence_type="readme_described",
        is_fork=False,
        last_commit_at=NOW - dt.timedelta(days=10),
        commits_authored=20,
        now=NOW,
    )
    confident = compute_weight(confidence=0.9, **kwargs)
    unsure = compute_weight(confidence=0.2, **kwargs)
    assert unsure < confident
