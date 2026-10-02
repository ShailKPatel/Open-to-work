FROM python:3.12-slim

WORKDIR /app

# system deps: build tools only where a wheel isn't available (sentence-transformers
# pulls in torch, which ships wheels for this arch on PyPI, no compiler needed,
# but curl is handy for healthchecks / debugging, and is also how tectonic gets
# installed below). libgraphite2-3 is the one shared library tectonic needs
# that this base image lacks (per `ldd`); everything else it links against
# (libstdc++, libm, libgcc_s, libc) is already present.
RUN apt-get update && apt-get install -y --no-install-recommends curl libgraphite2-3 \
    && rm -rf /var/lib/apt/lists/*

# Tectonic (app/resume_build/compile.py): a self-contained LaTeX engine, one
# binary that fetches only the packages a document actually needs and caches
# them, chosen over a full TeX Live install (multiple GB) for the same
# image-size reason as the CPU-only torch install below. The installer script
# drops the binary in the current directory, so it's moved to /usr/local/bin
# (already on PATH) and made executable explicitly.
RUN curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh \
    && mv tectonic /usr/local/bin/tectonic \
    && chmod +x /usr/local/bin/tectonic

COPY pyproject.toml ./

# CPU-only torch first, from PyTorch's own CPU wheel index. PyPI's default
# torch wheel drags in the full CUDA/GPU stack as separate packages
# (nvidia_cudnn, nvidia_cublas, nccl, triton...) even though nothing here
# uses a GPU: several GB and 5+ minutes of downloads for nothing. Installing
# the CPU build first satisfies sentence-transformers' torch dependency
# without pip ever reaching for the GPU one.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# -e install needs the package source present, but only app/__init__.py-level
# metadata; a placeholder empty app/ satisfies it so this layer (and the torch
# layer above) stay cached across ordinary code edits. The real app/ is
# copied in below, after deps are settled.
RUN mkdir app && touch app/__init__.py
RUN pip install --no-cache-dir -e ".[dev]"

# Playwright's Chromium, for app/ingest/jobs/auth_fetch.py (login-walled
# job posting fetch, real automated browser login). --with-deps also
# installs the OS-level shared libraries Chromium needs on this base image
# (fonts, audio/video codec stubs, etc), delegated to Playwright's own
# installer since that list is long and maintained per Chromium version.
# Adds roughly 300MB to the image.
RUN playwright install --with-deps chromium

# Fill Tectonic's package cache now, so a resume build never downloads
# LaTeX packages or fonts mid-compile (a cold fetch can outlast
# compile.py's timeout). Only app/resume_build/ is copied first, so this
# layer reruns when templates change, not on every code edit.
COPY app/resume_build ./app/resume_build
RUN python -m app.resume_build.warm_tectonic

# Code changes invalidate only from here down; deps above stay cached.
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
