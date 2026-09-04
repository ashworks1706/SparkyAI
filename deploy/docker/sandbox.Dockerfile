# syntax=docker/dockerfile:1.7
# The venv is built on the runtime base so its interpreter path stays valid.
FROM mcr.microsoft.com/playwright/python:v1.50.0-noble
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY apps/sandbox/pyproject.toml apps/sandbox/.python-version ./
RUN uv sync --no-dev --no-install-project
COPY apps/sandbox .
RUN uv sync --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8090
ENTRYPOINT ["sandbox"]
CMD ["serve"]
