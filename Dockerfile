# syntax=docker/dockerfile:1

# Every file fetched at build time is pinned to a version and checked
# against its SHA-256, so a changed upstream file fails the build instead
# of landing in the image. Tailwind and Tectonic ship one binary per CPU
# architecture, hence one download stage per arch (TARGETARCH is set by
# BuildKit).

FROM python:3.12-slim AS downloads-amd64
ADD --chmod=755 --checksum=sha256:4af3198c015616ea7d6617974ec3d70d987ecc00c1ca8463b0a30fd65cc7c06e \
    https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.19/tailwindcss-linux-x64 /tools/tailwindcss
ADD --checksum=sha256:1a715688baf591e650c8aeb160ae934e181685eecbb38b317de30b269ac5d606 \
    https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-gnu.tar.gz /tools/tectonic.tar.gz

FROM python:3.12-slim AS downloads-arm64
ADD --chmod=755 --checksum=sha256:e5b2d27694daa80cc52ec29553ba2c6bd43d86bd51a9d633ed24058b9c05a676 \
    https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.19/tailwindcss-linux-arm64 /tools/tailwindcss
ADD --checksum=sha256:b10954a95404f3ab2328d2fa59a5ebab8e657f893fab096f98be8db7c0c979b8 \
    https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-aarch64-unknown-linux-musl.tar.gz /tools/tectonic.tar.gz

FROM downloads-${TARGETARCH} AS tools
RUN tar -xzf /tools/tectonic.tar.gz -C /tools && rm /tools/tectonic.tar.gz

# Pages load nothing from a CDN: Tailwind is compiled to one stylesheet
# holding only the classes the templates use, and the JS libraries are
# served by the app itself.
FROM tools AS assets
WORKDIR /build
COPY app/web/tailwind.config.js app/web/tailwind.css ./
COPY app/web/templates ./templates
RUN mkdir -p /static/vendor \
    && /tools/tailwindcss -c tailwind.config.js -i tailwind.css -o /static/app.css --minify
ADD --chmod=644 --checksum=sha256:232519394c6c8fdba6f362b1d9da16106db513cdbf899011f00daab4051df31c \
    https://cdn.jsdelivr.net/npm/alpinejs@3.17.4/dist/cdn.min.js /static/vendor/alpine.min.js
ADD --chmod=644 --checksum=sha256:15fabce5b65898b32b03f5ed25e9f891a729ad4c0d6d877110a7744aa847a894 \
    https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js /static/vendor/marked.min.js
ADD --chmod=644 --checksum=sha256:2c90a9b46d6463f26038a29b686e82bc91de01fdac9d5229e7cfe3b360134ea2 \
    https://cdn.jsdelivr.net/npm/dompurify@3.4.16/dist/purify.min.js /static/vendor/purify.min.js

FROM python:3.12-slim

WORKDIR /app

# system deps: build tools only where a wheel isn't available (sentence-transformers
# pulls in torch, which ships wheels for this arch on PyPI, no compiler needed,
# but curl is handy for healthchecks / debugging). libgraphite2-3 is the one
# shared library tectonic needs that this base image lacks (per `ldd`);
# everything else it links against (libstdc++, libm, libgcc_s, libc) is
# already present.
RUN apt-get update && apt-get install -y --no-install-recommends curl libgraphite2-3 \
    && rm -rf /var/lib/apt/lists/*

# Tectonic (app/resume_build/compile.py): a self-contained LaTeX engine, one
# binary that fetches only the packages a document actually needs and caches
# them, chosen over a full TeX Live install (multiple GB) for the same
# image-size reason as the CPU-only torch install below.
COPY --from=tools /tools/tectonic /usr/local/bin/tectonic

COPY pyproject.toml constraints.txt ./

# CPU-only torch first, from PyTorch's own CPU wheel index. PyPI's default
# torch wheel drags in the full CUDA/GPU stack as separate packages
# (nvidia_cudnn, nvidia_cublas, nccl, triton...) even though nothing here
# uses a GPU: several GB and 5+ minutes of downloads for nothing. Installing
# the CPU build first satisfies sentence-transformers' torch dependency
# without pip ever reaching for the GPU one. PyPI is an extra index for
# torch's own dependencies at their pinned versions (the CPU index does not
# carry them); torch still comes from the CPU index, as 2.x+cpu sorts above 2.x.
RUN pip install --no-cache-dir -c constraints.txt torch \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# -e install needs the package source present, but only app/__init__.py-level
# metadata; a placeholder empty app/ satisfies it so this layer (and the torch
# layer above) stay cached across ordinary code edits. The real app/ is
# copied in below, after deps are settled.
RUN mkdir app && touch app/__init__.py
# constraints.txt pins every version, so a rebuild installs exactly what CI
# tested rather than whatever is newest that day.
RUN pip install --no-cache-dir -c constraints.txt -e .

# Fill Tectonic's package cache now, so a resume build never downloads
# LaTeX packages or fonts mid-compile (a cold fetch can outlast
# compile.py's timeout). Only app/resume_build/ is copied first, so this
# layer reruns when templates change, not on every code edit. The
# downloads go to a BuildKit cache mount first, so a build cut off by a
# slow or failing package server keeps what it fetched and the next
# attempt resumes instead of starting over.
COPY app/resume_build ./app/resume_build
RUN --mount=type=cache,id=tectonic-warm,target=/tmp/tectonic-cache \
    TECTONIC_CACHE_DIR=/tmp/tectonic-cache python -m app.resume_build.warm_tectonic \
    && mkdir -p /root/.cache && cp -a /tmp/tectonic-cache /root/.cache/tectonic

# Code changes invalidate only from here down; deps above stay cached.
COPY app ./app
COPY --from=assets /static ./app/web/static

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
