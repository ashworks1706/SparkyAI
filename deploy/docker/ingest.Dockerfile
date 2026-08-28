# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY apps/ingest/pyproject.toml apps/ingest/.python-version ./
RUN uv sync --no-dev --no-install-project
COPY apps/ingest .
RUN uv sync --no-dev

FROM mcr.microsoft.com/playwright/python:v1.50.0-noble
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["sparky-ingest"]
CMD ["schedule"]
