# syntax=docker/dockerfile:1.7
# The scraper: ingestion and live query worker.
FROM mcr.microsoft.com/playwright/python:v1.50.0-noble
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY apps/scraper/pyproject.toml apps/scraper/.python-version ./
RUN uv sync --no-dev --no-install-project
COPY apps/scraper .
# Settings file; environment variables override it.
COPY sparky.toml /app/sparky.toml
RUN uv sync --no-dev
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["scraper"]
