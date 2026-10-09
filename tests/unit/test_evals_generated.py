import yaml

from app.evals import generated


def _files(tmp_path):
    records = [
        {"id": "a#0", "split": "test", "evidence": "e", "bullet": "b0", "passed_filter": True},
        {"id": "a#1", "split": "test", "evidence": "e", "bullet": "b1", "passed_filter": False},
        {"id": "b#0", "split": "dev", "evidence": "e", "bullet": "b2", "passed_filter": True},
        {"id": "c#0", "split": "test", "evidence": "e", "bullet": "b3", "passed_filter": True},
    ]
    labels = {
        "a#0": {"grounded": True},
        "a#1": {"grounded": False, "claim": "x"},
        "b#0": {"grounded": False, "borderline": True},
    }
    bullets = tmp_path / "bullets.yaml"
    label_file = tmp_path / "labels.yaml"
    bullets.write_text(yaml.safe_dump(records))
    label_file.write_text(yaml.safe_dump(labels))
    return bullets, label_file


def test_load_keeps_labeled_bullets_of_one_split(tmp_path):
    bullets, labels = _files(tmp_path)
    test = generated.load_generated("test", bullets, labels)
    assert [b.id for b in test] == ["a#0", "a#1"]
    both = generated.load_generated(None, bullets, labels)
    assert [b.id for b in both] == ["a#0", "a#1", "b#0"]
    assert both[2].kind == "borderline" and both[0].kind == "clear"


def test_filter_score_counts_each_cell(tmp_path):
    bullets, labels = _files(tmp_path)
    score = generated.score_filter(generated.load_generated(None, bullets, labels))
    assert score.kept_grounded == 1
    assert score.dropped_ungrounded == 1
    assert score.kept_ungrounded == 1
    assert score.dropped_grounded == 0
    assert score.total == 3 and score.ungrounded == 2


def test_judge_runs_on_the_stored_evidence(tmp_path, monkeypatch):
    bullets, labels = _files(tmp_path)
    seen = []

    def fake_judge(evidence, bullet, account_id):
        seen.append((evidence, bullet, account_id))
        return bullet == "b0"

    monkeypatch.setattr("app.evals.groundedness.judge_bullet", fake_judge)
    score = generated.run_judge_generated(generated.load_generated("test", bullets, labels))
    assert seen == [("e", "b0", None), ("e", "b1", None)]
    assert score.accuracy == 1.0
