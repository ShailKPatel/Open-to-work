"""What an uploaded file actually is, read from its first bytes.

The type a browser declares for an upload comes from the file name, which
the person (or a page posting to the app) chooses freely. Anything that
decides how a file is served back, or which reader it goes to, uses these
instead.
"""

from __future__ import annotations

PDF = "application/pdf"


def sniff_image_type(data: bytes) -> str | None:
    """PNG, JPEG, GIF or WebP by signature, else None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def sniff_type(data: bytes) -> str | None:
    """A PDF or one of the images above, else None."""
    if data.startswith(b"%PDF-"):
        return PDF
    return sniff_image_type(data)


def sniff_file(path: str) -> str | None:
    """sniff_type() for a file on disk, reading only its first bytes."""
    with open(path, "rb") as f:
        return sniff_type(f.read(16))
