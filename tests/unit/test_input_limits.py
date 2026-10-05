import io

import pytest
from fastapi import HTTPException

from app.api import input_limits
from app.api.input_limits import (
    LARGE_INPUT_CODE,
    file_notes,
    files_notes,
    require_confirmation,
    text_notes,
)


def _pdf(pages: int) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_inputs_under_every_limit_need_no_confirmation():
    assert file_notes(_pdf(2)) == []
    assert files_notes([b"x" * 1000, b"y" * 1000]) == []
    assert text_notes("A normal job posting.") == []
    require_confirmation("file", [], confirmed=False)


def test_each_limit_names_what_it_found(monkeypatch):
    monkeypatch.setattr(input_limits, "SOFT_MAX_FILE_MB", 0.5)
    monkeypatch.setattr(input_limits, "SOFT_MAX_PDF_PAGES", 3)
    monkeypatch.setattr(input_limits, "SOFT_MAX_TEXT_CHARS", 10)

    assert file_notes(_pdf(4)) == ["4 pages"]
    assert file_notes(b"x" * (2 * 1024 * 1024)) == ["2 MB"]
    assert files_notes([b"x" * (400 * 1024), b"y" * (400 * 1024)]) == ["1 MB"]
    assert text_notes("x" * 12_345) == ["12,345 characters"]


def test_a_file_that_is_not_a_readable_pdf_is_only_sized():
    assert file_notes(b"%PDF-1.4 broken") == []
    assert file_notes(b"\x89PNG not a pdf") == []


def test_over_a_limit_asks_unless_already_confirmed():
    with pytest.raises(HTTPException) as caught:
        require_confirmation("file", ["14 MB", "40 pages"], confirmed=False)

    assert caught.value.status_code == 409
    assert caught.value.detail == {
        "code": LARGE_INPUT_CODE,
        "message": "This file is 14 MB / 40 pages and will use a lot of tokens. "
        "Process it anyway?",
    }
    require_confirmation("file", ["14 MB"], confirmed=True)
