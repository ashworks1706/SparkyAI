# syntax=docker/dockerfile:1.7
# Ingestion and live query worker; venv built on runtime base image.
FROM mcr.microsoft.com/playwright/python:v1.50.0-noble
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY apps/scraper/pyproject.toml apps/scraper/.python-version ./
RUN uv sync --no-dev --no-install-project
COPY apps/scraper .
# Settings layer at /app, overridden by environment.
COPY sparky.toml /app/sparky.toml
RUN uv sync --no-dev
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["scraper"]
