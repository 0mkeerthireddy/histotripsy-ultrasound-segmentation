# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Histotripsy ablation segmentation: multi-stage, production-oriented image.
#
# Stages:  base -> builder -> test -> runtime
#   base     shared Python settings
#   builder  creates a virtualenv with pinned CPU-only dependencies
#   test     runs the unit tests (build with --target test in CI)
#   runtime  minimal final image: venv + source only, non-root user
#
# Build:   docker build -t histotripsy-seg:1.0.0 --target runtime .
# Test:    docker build --target test .
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.11
# For fully reproducible builds, pin the base image by digest:
#   python:3.11-slim-bookworm@sha256:<digest>
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim-bookworm

# ---------------------------- base -----------------------------------------
FROM ${BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---------------------------- builder --------------------------------------
FROM base AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .

# 1) CPU-only PyTorch (much smaller than the default CUDA build), version taken
#    from requirements.txt so there is a single source of truth.
# 2) Remaining pinned dependencies from PyPI.
# The pip cache is a BuildKit cache mount, so it never ends up in an image layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        "$(grep -E '^torch==' requirements.txt)" \
    && pip install -r requirements.txt

# ---------------------------- test -----------------------------------------
FROM builder AS test

COPY requirements-dev.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

# The build fails here if any test fails.
RUN python -m pytest -q

# ---------------------------- runtime --------------------------------------
FROM base AS runtime

ARG VERSION=dev
ARG VCS_REF=unknown
ARG APP_UID=10001

LABEL org.opencontainers.image.title="histotripsy-ablation-segmentation" \
      org.opencontainers.image.description="U-Net segmentation of histotripsy ablation zones in ultrasound images" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Unprivileged user with no login shell.
RUN groupadd --gid "${APP_UID}" app \
    && useradd --uid "${APP_UID}" --gid app --create-home \
       --home-dir /home/app --shell /usr/sbin/nologin app

# Only the virtualenv and application code are copied: no compilers, no pip
# cache, no test tooling.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app

COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app configs ./configs

# Mount points for data, checkpoints and predictions (mounted at runtime).
RUN mkdir -p /app/data /app/runs /app/preds \
    && chown app:app /app/data /app/runs /app/preds

USER app

# This is a batch/CLI workload (train, evaluate, predict), not a long-running
# server, so there is no HEALTHCHECK. Add one if you later wrap it in an API.
ENTRYPOINT ["python", "-m"]
CMD ["src.predict", "--help"]
