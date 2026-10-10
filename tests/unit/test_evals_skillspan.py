"""app/evals/skillspan.py: BIO spans, posting assembly, exact-name labels."""

import json

from app.evals.skillspan import load_postings, relevant_skills, spans


def test_spans_reads_bio_tags():
    tokens = ["Use", "Ruby", "on", "Rails", "and", "Go", "daily"]
    tags = ["O", "B", "I", "I", "O", "B", "O"]

    assert spans(tokens, tags) == ["Ruby on Rails", "Go"]


def test_an_inside_tag_without_a_begin_is_ignored():
    assert spans(["a", "b"], ["I", "O"]) == []


def test_load_postings_groups_tech_sentences_by_document(tmp_path):
    rows = {
        "train": [
            {"idx": 1, "tokens": ["Senior", "dev"], "tags_knowledge": ["O", "O"],
             "tags_skill": ["O", "O"], "source": "tech"},
            {"idx": 1, "tokens": ["Know", "Python"], "tags_knowledge": ["O", "B"],
             "tags_skill": ["O", "O"], "source": "tech"},
            {"idx": 2, "tokens": ["Excel"], "tags_knowledge": ["B"],
             "tags_skill": ["O"], "source": "house"},
        ],
        "dev": [],
        "test": [],
    }
    for split, items in rows.items():
        (tmp_path / f"{split}.json").write_text("\n".join(json.dumps(r) for r in items))

    postings = load_postings(tmp_path)

    assert list(postings) == ["skillspan-train-1"]
    assert postings["skillspan-train-1"]["text"] == "Senior dev\nKnow Python"
    assert postings["skillspan-train-1"]["knowledge"] == {"python"}


def test_relevant_skills_match_whole_spans_only():
    knowledge = {"python", "react native", "aws"}

    assert relevant_skills(knowledge, ["Python", "React", "AWS", "Go"]) == ["AWS", "Python"]


def test_rule_v2_reads_the_parts_of_a_compound_span():
    knowledge = {"core-java/spring/spring-boot", "angular/react.js"}

    found = relevant_skills(knowledge, ["Java", "Spring Boot", "React", "Angular", "Go"])

    assert found == ["Angular", "Java", "React", "Spring Boot"]


def test_rule_v2_compares_whole_parts_never_substrings():
    assert relevant_skills({"javascript"}, ["Java", "JavaScript"]) == ["JavaScript"]
    assert relevant_skills({"c++"}, ["C", "C++"]) == ["C++"]


def test_rule_v1_needs_the_whole_span():
    knowledge = {"angular/react.js", "python"}

    assert relevant_skills(knowledge, ["React", "Python"], rule="v1") == ["Python"]
