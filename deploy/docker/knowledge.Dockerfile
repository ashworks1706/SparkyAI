# syntax=docker/dockerfile:1.7
# One image for the knowledge service; entrypoint selects knowledge-api or knowledge-scraper.
# The venv is built on the runtime base so its interpreter path stays valid.
FROM mcr.microsoft.com/playwright/python:v1.50.0-noble
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY apps/knowledge/pyproject.toml apps/knowledge/.python-version ./
RUN uv sync --no-dev --no-install-project
COPY apps/knowledge .
RUN uv sync --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8081
ENTRYPOINT ["knowledge-api"]
