from app.evals.golden import GoldenPair, load_golden_set, save_golden_set, upsert_pair


def _pair(**overrides) -> GoldenPair:
    defaults = dict(
        id="1-skill_evidence-1", account_id=1, collection="skill_evidence",
        query_text="backend engineer, python", relevant_ids=[1, 2],
    )
    defaults.update(overrides)
    return GoldenPair(**defaults)


def test_load_missing_file_returns_empty_list(tmp_path):
    assert load_golden_set(tmp_path / "nope.yaml") == []


def test_save_and_load_round_trips(tmp_path):
    path = tmp_path / "golden_set.yaml"
    save_golden_set([_pair()], path)

    loaded = load_golden_set(path)

    assert len(loaded) == 1
    assert loaded[0].id == "1-skill_evidence-1"
    assert loaded[0].relevant_ids == [1, 2]


def test_upsert_replaces_same_id_instead_of_duplicating():
    pairs = [_pair(relevant_ids=[1])]
    updated = upsert_pair(pairs, _pair(relevant_ids=[1, 2, 3]))

    assert len(updated) == 1
    assert updated[0].relevant_ids == [1, 2, 3]


def test_upsert_appends_a_new_id():
    pairs = [_pair(id="1-skill_evidence-1")]
    updated = upsert_pair(pairs, _pair(id="1-skill_evidence-2"))

    assert len(updated) == 2
    assert {p.id for p in updated} == {"1-skill_evidence-1", "1-skill_evidence-2"}


def test_saved_file_is_human_readable_yaml(tmp_path):
    path = tmp_path / "golden_set.yaml"
    save_golden_set([_pair()], path)

    text = path.read_text()
    assert "query_text" in text
    assert "backend engineer" in text
