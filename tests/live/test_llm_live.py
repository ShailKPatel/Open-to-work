"""Real, billed LLM API calls. Not part of `make test` (pyproject.toml's
testpaths is ["tests/unit"]); run with `make test-live`, or
`pytest tests/live -v -s` (`-s` keeps the printed cost/token numbers).

Skipped unless LIVE_LLM_API_KEY is set, so a key in the environment never
turns on paid calls in the normal test run. LIVE_LLM_PROVIDER picks which
provider the key belongs to (default "gemini"); set LIVE_LLM_BULK_MODEL and
LIVE_LLM_QUALITY_MODEL to matching "<provider>/<model>" strings when using a
different provider.

Covers text-only, image, PDF, and multi-turn context through
`core/llm.py`'s single entrypoint, against a real provider rather than the
injected fakes the unit tests use.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests

import app.core.db as db_module
from app.core import api_keys_store
from app.core.app_settings import update_llm_settings
from app.core.db import init_db
from app.core.llm import complete, user_message
from app.core.settings import get_settings

_API_KEY = os.environ.get("LIVE_LLM_API_KEY", "")
_PROVIDER = os.environ.get("LIVE_LLM_PROVIDER", "gemini")

pytestmark = pytest.mark.skipif(
    not _API_KEY,
    reason="LIVE_LLM_API_KEY not set; live tests are opt-in (real billed calls)",
)


def _reset_db(tmp_path: Path):
    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()
    row, detail = api_keys_store.add_key(_PROVIDER, "live test", {"api_key": _API_KEY}, None)
    assert row is not None, detail
    update_llm_settings(
        bulk_model=os.environ.get("LIVE_LLM_BULK_MODEL") or None,
        quality_model=os.environ.get("LIVE_LLM_QUALITY_MODEL") or None,
    )


def _report(label: str, result) -> None:
    print(
        f"\n[live:{label}] model={result.model} cached={result.cached} "
        f"tokens_in={result.tokens_in} tokens_out={result.tokens_out} "
        f"cost_usd=${result.cost_usd:.6f}\n"
        f"[live:{label}] content={result.content!r}"
    )


def _build_minimal_pdf(text: str) -> bytes:
    """Hand-built single-page PDF with correct xref byte offsets, so no
    reportlab/fpdf dependency is needed just to exercise PDF input.
    """
    content = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 150] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = "%PDF-1.4\n"
    body = ""
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(header.encode("latin-1")) + len(body.encode("latin-1")))
        body += f"{i} 0 obj\n{obj}\nendobj\n"
    xref_offset = len(header.encode("latin-1")) + len(body.encode("latin-1"))
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
    )
    return (header + body + xref + trailer).encode("latin-1")


def test_text_completion(tmp_path):
    _reset_db(tmp_path)
    result = complete("bulk", [user_message("Reply with exactly one word: PONG")])
    _report("text", result)
    assert "pong" in result.content.lower()
    assert result.cost_usd >= 0.0


_WIKI_HEADERS = {
    # Wikimedia requires a descriptive User-Agent per their robot policy
    # (https://w.wiki/4wJS); an unset or generic one gets a 403.
    "User-Agent": "open-to-work-test/1.0 (https://github.com/; test suite, low volume)"
}


def test_image_completion_real_dog_photo(tmp_path):
    _reset_db(tmp_path)
    # fetched dynamically (Wikipedia's current "Dog" article thumbnail)
    # rather than a hardcoded upload.wikimedia.org URL, which rots
    summary = requests.get(
        "https://en.wikipedia.org/api/rest_v1/page/summary/Dog",
        headers=_WIKI_HEADERS,
        timeout=10,
    ).json()
    image_url = summary["thumbnail"]["source"]
    image_bytes = requests.get(image_url, headers=_WIKI_HEADERS, timeout=10).content

    result = complete(
        "quality",
        [user_message("What animal is in this image? Answer with one word.", images=[image_bytes])],
    )
    _report("image", result)
    assert "dog" in result.content.lower()


def test_pdf_completion(tmp_path):
    _reset_db(tmp_path)
    marker = "OPENTOWORK-MARKER-73921"
    pdf_bytes = _build_minimal_pdf(marker)

    result = complete(
        "quality",
        [
            user_message(
                "This PDF contains one distinctive marker string. Reply with "
                "just that marker string, nothing else.",
                files=[pdf_bytes],
            )
        ],
    )
    _report("pdf", result)
    assert marker in result.content


def test_multi_turn_context_carries_across_calls(tmp_path):
    _reset_db(tmp_path)
    messages = [user_message("My favorite number is 42. Just say OK.")]
    first = complete("bulk", messages)
    _report("context-turn-1", first)

    messages.append({"role": "assistant", "content": first.content})
    messages.append(user_message("What is my favorite number? Reply with just the digits."))
    second = complete("bulk", messages)
    _report("context-turn-2", second)

    assert "42" in second.content
