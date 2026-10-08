"""scripts/check_eval_baseline.py's comparison: the app retrieval always, and each
candidate list the baseline has numbers for."""

import pytest

from scripts.check_eval_baseline import _rows

_SCORES = {"precision_at_5": 0.5, "recall_at_10": 0.9}


def _report(candidates: dict | None = None) -> dict:
    return {"retrieval": _SCORES, "bm25": _SCORES, "candidates": candidates or {}}


def _baseline(candidates: dict | None = None) -> dict:
    baseline = {"retrieval": _SCORES, "tolerance": 0.01}
    if candidates is not None:
        baseline["candidates"] = candidates
    return baseline


def test_retrieval_only_baseline_gates_retrieval_only():
    rows = _rows(_report({"skills": _SCORES}), _baseline())

    assert [row[0] for row in rows] == ["precision@5", "recall@10"]
    assert all(row[4] for row in rows)


def test_candidate_drop_beyond_tolerance_fails_that_row():
    dropped = {"precision_at_5": 0.45, "recall_at_10": 0.9}
    rows = _rows(_report({"skills": dropped}), _baseline({"skills": _SCORES}))

    verdicts = {row[0]: row[4] for row in rows}
    assert verdicts["candidate skills precision@5"] is False
    assert verdicts["candidate skills recall@10"] is True
    assert verdicts["precision@5"] is True


def test_report_missing_a_baselined_candidate_list_is_an_error():
    with pytest.raises(SystemExit, match="no candidate projects"):
        _rows(_report(), _baseline({"projects": _SCORES}))
