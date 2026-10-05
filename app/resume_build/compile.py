"""Compiles a rendered .tex string (app/resume_build/latex.py's
render_resume() output) into PDF bytes via Tectonic
(tectonic-typesetting.github.io), a self-contained LaTeX engine: one
static binary, no multi-gigabyte TeX Live install, fetches only the
packages a given document actually needs and caches them. Chosen over
plain pdflatex/TeX Live for that image-size reason.

_run_fn is an injection point for tests, same pattern as
app/core/llm.py's _completion_fn and app/core/embeddings.py's _encode_fn:
production callers never pass it, it defaults to subprocess.run. Tectonic
is usually not installed on a dev host, so every test here mocks the subprocess
call; the Dockerfile installs the real binary for the one place this
actually runs against real input.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_DEFAULT_TIMEOUT_SECONDS = 60


class TectonicNotInstalledError(RuntimeError):
    """No `tectonic` binary on PATH. Distinct from CompileError (a real
    compile attempt that failed) so a caller can tell "this environment
    isn't set up for PDF generation at all" apart from "this specific
    document has a LaTeX error."""


class CompileError(RuntimeError):
    """tectonic ran and reported failure (bad LaTeX, missing package it
    couldn't fetch, timeout). Carries tectonic's own stderr/stdout so the
    caller has something real to show or log, not just a bare exit code.
    """


def compile_tex(
    tex_source: str, timeout: int = _DEFAULT_TIMEOUT_SECONDS, _run_fn: Any = None
) -> bytes:
    """Writes tex_source to a throwaway temp directory, runs tectonic
    against it, returns the resulting PDF's bytes. Nothing about the
    output persists after this call: callers that want the PDF saved
    (app/api/, a future storage step) do that themselves with the
    returned bytes.
    """
    run = _run_fn or subprocess.run

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        tex_path = tmp / "resume.tex"
        tex_path.write_text(tex_source, encoding="utf-8")

        try:
            # Shell escape is off by default in Tectonic; untrusted mode
            # also turns off anything else it knows to be unsafe, in case
            # resume text ever got past escape_latex().
            result = run(
                ["tectonic", "--outdir", str(tmp), str(tex_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "TECTONIC_UNTRUSTED_MODE": "1"},
            )
        except FileNotFoundError as e:
            raise TectonicNotInstalledError(
                "tectonic is not installed or not on PATH; see the Dockerfile for the "
                "expected install step"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise CompileError(f"tectonic timed out after {timeout}s") from e

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip() or "no output captured"
            raise CompileError(f"tectonic exited {result.returncode}: {detail}")

        pdf_path = tmp / "resume.pdf"
        if not pdf_path.exists():
            raise CompileError("tectonic reported success but produced no resume.pdf")
        return pdf_path.read_bytes()
