# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.
#
# Build with the directory that contains pyproject.toml and uv.lock as the logical "env root":
#   cd envs/email_triage_env && docker build -f server/Dockerfile -t email-triage-env .
# From a monorepo root (context = repo root), copy only this package:
#   docker build -f envs/email_triage_env/server/Dockerfile --build-arg SOURCE_DIR=envs/email_triage_env -t email-triage-env .

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
# Documented for openenv / CI; image runtime does not consume it today
ARG ENV_NAME=email_triage_env

# Path within build context to the env package (contains pyproject.toml). Default "." for standalone / openenv build.
ARG SOURCE_DIR=.
COPY ${SOURCE_DIR} /app/env

# For in-repo builds, openenv is already vendored in the build context
# For standalone builds, openenv will be installed via pyproject.toml
WORKDIR /app/env

# Host .venv may be copied if build context lacks a matching .dockerignore; uv refuses invalid trees.
RUN rm -rf .venv

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
    
# Install project + locked deps (--no-install-project breaks this setuptools package-dir layout).
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Source tree on path: supports flat `server.app` / `models` fallbacks in code; the
# installed package (below) is the primary import path at runtime.
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check (requires `curl` in the runtime base image)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI via the installed package (matches pyproject `server` script and `python -m email_triage_env.server.app`)
WORKDIR /app/env
CMD ["uvicorn", "email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
