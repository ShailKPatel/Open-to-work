"""app/resume_build/compile.py's compile_tex(). subprocess.run is
injected (_run_fn), same pattern as app/core/llm.py's _completion_fn. Tectonic is usually
not installed on a dev host, so only the wrapper logic is tested here.
"""

import subprocess
from pathlib import Path

import pytest

from app.resume_build.compile import CompileError, TectonicNotInstalledError, compile_tex


def test_missing_binary_raises_typed_error():
    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("no such file: tectonic")

    with pytest.raises(TectonicNotInstalledError):
        compile_tex(
            r"\documentclass{article}\begin{document}x\end{document}", _run_fn=_raise_not_found
        )


def test_nonzero_exit_raises_compile_error_with_stderr():
    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="! Undefined control sequence."
        )

    with pytest.raises(CompileError, match="Undefined control sequence"):
        compile_tex(r"\badcommand", _run_fn=_fake_run)


def test_timeout_raises_compile_error():
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="tectonic", timeout=1)

    with pytest.raises(CompileError, match="timed out"):
        compile_tex("x", timeout=1, _run_fn=_raise_timeout)


def test_success_but_no_pdf_raises_compile_error():
    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    with pytest.raises(CompileError, match="no resume.pdf"):
        compile_tex("x", _run_fn=_fake_run)


def test_success_returns_pdf_bytes():
    def _fake_run(cmd, **kwargs):
        outdir = Path(cmd[2])
        (outdir / "resume.pdf").write_bytes(b"%PDF-1.4 fake pdf bytes")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    result = compile_tex(
        r"\documentclass{article}\begin{document}x\end{document}", _run_fn=_fake_run
    )

    assert result == b"%PDF-1.4 fake pdf bytes"


def test_uses_tectonic_and_the_written_tex_path():
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Read the written .tex now, inside the call: compile_tex's temp
        # directory is gone by the time this function returns.
        captured["tex_content"] = Path(cmd[3]).read_text()
        outdir = Path(cmd[2])
        (outdir / "resume.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    compile_tex("x", _run_fn=_fake_run)

    assert captured["cmd"][0] == "tectonic"
    assert captured["cmd"][1] == "--outdir"
    assert captured["cmd"][3].endswith("resume.tex")
    assert captured["tex_content"] == "x"


def test_tectonic_runs_in_untrusted_mode():
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env") or {}
        (Path(cmd[2]) / "resume.pdf").write_bytes(b"%PDF")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    compile_tex("x", _run_fn=_fake_run)

    assert captured["env"]["TECTONIC_UNTRUSTED_MODE"] == "1"
    # Shell escape is never asked for on the command line either.
    assert "shell-escape" not in " ".join(captured["cmd"])
