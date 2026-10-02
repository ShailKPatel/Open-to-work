import datetime as dt
import json

import pytest

from app.profile.month_year import newest_first, normalize, normalize_or_none, sort_key


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (dt.date(2026, 3, 15), "Mar 2026"),
        ("2026-03-01", "Mar 2026"),
        ("2026-03", "Mar 2026"),
        ("2026/3", "Mar 2026"),
        ("03/2026", "Mar 2026"),
        ("3-2026", "Mar 2026"),
        ("Mar 2026", "Mar 2026"),
        ("mar 2026", "Mar 2026"),
        ("MAR 2026", "Mar 2026"),
        ("March 2026", "Mar 2026"),
        ("Mar. 2026", "Mar 2026"),
        ("Sept 2021", "Sep 2021"),
        ("December, 2020", "Dec 2020"),
        ("  2021 ", "2021"),
        ("", None),
        (None, None),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", ["soon", "Foo 2026", "13/2026", "2026-00", "1820", "present"])
def test_normalize_rejects_unreadable(raw):
    with pytest.raises(ValueError):
        normalize(raw)
    assert normalize_or_none(raw) is None


def test_sort_key_orders_by_month_with_blanks_first():
    values = ["2021", "Mar 2026", None, "Jan 2026", "Dec 2021"]
    assert sorted(values, key=sort_key) == [None, "2021", "Dec 2021", "Jan 2026", "Mar 2026"]


def test_newest_first_puts_undated_last():
    class Row:
        def __init__(self, start_date):
            self.start_date = start_date

    rows = [Row(None), Row("Aug 2019"), Row("Feb 2024"), Row("2022")]
    assert [r.start_date for r in newest_first(rows)] == ["Feb 2024", "2022", "Aug 2019", None]


def test_migration_rewrites_iso_dates_and_resume_snapshots(tmp_path):
    from sqlalchemy import create_engine, text

    from app.core.db.migrations import _migrate_month_year_dates

    engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    with engine.connect() as conn:
        for table in ("experiences", "education"):
            conn.execute(
                text(
                    f"CREATE TABLE {table} "
                    "(id INTEGER PRIMARY KEY, start_date DATE, end_date DATE)"
                )
            )
            conn.execute(
                text(f"INSERT INTO {table} VALUES (1, '2021-03-01', NULL), (2, 'Jun 2020', 'junk')")
            )
        conn.execute(
            text(
                "CREATE TABLE resumes (id INTEGER PRIMARY KEY, experiences_json JSON, "
                "education_json JSON)"
            )
        )
        conn.execute(
            text("INSERT INTO resumes VALUES (1, :e, '[]')"),
            {"e": json.dumps([{"company": "Acme", "start_date": "2022-08-01", "end_date": None}])},
        )
        conn.commit()

    _migrate_month_year_dates(engine)
    _migrate_month_year_dates(engine)  # idempotent

    with engine.connect() as conn:
        for table in ("experiences", "education"):
            rows = conn.execute(text(f"SELECT start_date, end_date FROM {table} ORDER BY id")).all()
            assert [tuple(r) for r in rows] == [("Mar 2021", None), ("Jun 2020", "junk")]
        snapshot = conn.execute(text("SELECT experiences_json FROM resumes")).scalar()
    assert json.loads(snapshot)[0]["start_date"] == "Aug 2022"
