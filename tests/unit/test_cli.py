"""Command-line entry points: `python -m app.ingest.github` (`make ingest`)
and `python -m app.evals` (`make eval`). The work they delegate to is
faked; these tests cover argument handling and what gets printed."""

import sys

import pytest

from app.evals import __main__ as evals_cli
from app.evals.run import MetricsReport
from app.ingest.github import __main__ as ingest_cli
from app.ingest.github.sync import SyncSummary


def _report(**overrides) -> MetricsReport:
    fields = dict(
        generated_at="2026-01-01T00:00:00Z",
        account_id=7,
        golden_set_size=4,
        pairs_scored=3,
        retrieval={"precision_at_5": 0.7, "recall_at_10": 0.9, "pairs_scored": 3},
        dense={"precision_at_5": 0.6, "recall_at_10": 0.8, "pairs_scored": 3},
        bm25={"precision_at_5": 0.4, "recall_at_10": 0.5, "pairs_scored": 3},
        retrieval_beats_bm25=True,
        precision_at_10=0.75,
        groundedness=0.9,
        groundedness_checked=10,
        cost_usd=0.0123,
        latency_ms_avg=120.0,
        llm_calls=10,
        notes=["two pairs had no labeled hits"],
    )
    fields.update(overrides)
    return MetricsReport(**fields)


def test_ingest_cli_without_username_prints_usage_and_exits_1(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["app.ingest.github"])

    with pytest.raises(SystemExit) as exc:
        ingest_cli.main()

    assert exc.value.code == 1
    assert "usage" in capsys.readouterr().out


def test_ingest_cli_initializes_db_then_syncs_the_given_username(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", ["app.ingest.github", "octocat"])
    monkeypatch.setattr(ingest_cli, "init_db", lambda: calls.append("init_db"))

    def fake_sync(username):
        calls.append(username)
        return SyncSummary(total_repos=3, fetched=2, cache_hits=1)

    monkeypatch.setattr(ingest_cli, "sync_account", fake_sync)

    ingest_cli.main()

    assert calls == ["init_db", "octocat"]
    assert "repos=3 fetched=2 cache_hits=1" in capsys.readouterr().out


def _run_evals_cli(monkeypatch, tmp_path, argv, report):
    calls = {}
    monkeypatch.setattr(sys, "argv", ["app.evals", *argv])
    monkeypatch.setattr(evals_cli, "init_db", lambda: None)

    def fake_run_eval(account_id, include_groundedness):
        calls["run_eval"] = (account_id, include_groundedness)
        return report

    monkeypatch.setattr(evals_cli, "run_eval", fake_run_eval)
    monkeypatch.setattr(evals_cli, "write_report", lambda r: tmp_path / "report.json")
    evals_cli.main()
    return calls


def test_evals_cli_full_run_prints_every_metric(monkeypatch, tmp_path, capsys):
    calls = _run_evals_cli(monkeypatch, tmp_path, ["7"], _report())
    out = capsys.readouterr().out

    assert calls["run_eval"] == (7, True)
    assert "Eval report for account 7" in out
    assert "retrieval beats the BM25 baseline" in out
    assert "retrieval (app)" in out and "dense (reference)" in out
    assert "precision@10 (vs ground truth): 0.750" in out
    assert "groundedness: 0.900 (10 bullets checked)" in out
    assert "cost this run: $0.0123" in out
    assert "two pairs had no labeled hits" in out
    assert f"Written to {tmp_path / 'report.json'}" in out


def test_evals_cli_no_groundedness_flag_and_unmeasured_metrics(monkeypatch, tmp_path, capsys):
    report = _report(
        pairs_scored=0,
        retrieval_beats_bm25=None,
        precision_at_10=None,
        groundedness=None,
        groundedness_checked=0,
        notes=[],
    )
    calls = _run_evals_cli(monkeypatch, tmp_path, ["7", "--no-groundedness"], report)
    out = capsys.readouterr().out

    assert calls["run_eval"] == (7, False)
    assert "retrieval vs baseline: not measured" in out
    assert "groundedness: not measured" in out
    assert "precision@10" not in out
    assert "Notes:" not in out


def test_evals_cli_reports_when_retrieval_loses_to_baseline(monkeypatch, tmp_path, capsys):
    _run_evals_cli(monkeypatch, tmp_path, ["7"], _report(retrieval_beats_bm25=False))

    assert "retrieval does NOT beat the BM25 baseline" in capsys.readouterr().out


def test_evals_cli_prints_paired_differences(monkeypatch, tmp_path, capsys):
    report = _report(
        differences={
            "retrieval - bm25": {"precision_at_5": {"mean": 0.1, "ci95": [0.02, 0.2]}},
            "retrieval - dense": {"precision_at_5": {"mean": 0.0, "ci95": None}},
        }
    )
    _run_evals_cli(monkeypatch, tmp_path, ["7"], report)
    out = capsys.readouterr().out

    assert "retrieval - bm25 precision_at_5: +0.100, 95% CI [+0.020, +0.200]" in out
    assert "retrieval - dense precision_at_5: +0.000, 95% CI n/a" in out


@pytest.mark.parametrize("argv", [[], ["not-a-number"]])
def test_evals_cli_rejects_missing_or_non_integer_account(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["app.evals", *argv])

    with pytest.raises(SystemExit) as exc:
        evals_cli.main()

    assert exc.value.code == 2
