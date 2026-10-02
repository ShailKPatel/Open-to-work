"""app/profile/job_place.py: work-mode buckets and city parsing."""

import pytest

from app.profile.job_place import cities, work_mode_kind


@pytest.mark.parametrize(
    ("mode", "location", "expected"),
    [
        ("Remote", "Bengaluru", "remote"),
        ("Hybrid", "Pune", "hybrid"),
        ("On-site", "Pune", "on_site"),
        ("", "Remote, India", "remote"),
        ("", "Work from home", "remote"),
        ("", "Bangalore (Hybrid)", "hybrid"),
        ("", "Pune", "unknown"),
        (None, None, "unknown"),
    ],
)
def test_work_mode_kind(mode, location, expected):
    assert work_mode_kind(mode, location) == expected


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("Pune, Maharashtra, India", ["Pune"]),
        ("Pune / Mumbai", ["Pune", "Mumbai"]),
        ("Pune or Mumbai", ["Pune", "Mumbai"]),
        ("Bangalore (Hybrid)", ["Bengaluru"]),
        ("Bengaluru / Bangalore", ["Bengaluru"]),
        ("Gurgaon, Haryana", ["Gurugram"]),
        ("Remote", []),
        ("Multiple locations", []),
        ("", []),
        (None, []),
    ],
)
def test_cities(location, expected):
    assert cities(location) == expected
