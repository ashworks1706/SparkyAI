# syntax=docker/dockerfile:1.7
# One image for the knowledge service; entrypoint selects knowledge-api or knowledge-scraper.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY apps/knowledge/pyproject.toml apps/knowledge/.python-version ./
RUN uv sync --no-dev --no-install-project
COPY apps/knowledge .
RUN uv sync --no-dev

FROM mcr.microsoft.com/playwright/python:v1.50.0-noble
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8081
ENTRYPOINT ["knowledge-api"]
