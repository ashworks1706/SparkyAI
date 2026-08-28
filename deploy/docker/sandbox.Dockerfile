# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY apps/sandbox/pyproject.toml apps/sandbox/.python-version ./
RUN uv sync --no-dev --no-install-project
COPY apps/sandbox .
RUN uv sync --no-dev

FROM mcr.microsoft.com/playwright/python:v1.50.0-noble
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8090
ENTRYPOINT ["sandbox"]
CMD ["serve"]
