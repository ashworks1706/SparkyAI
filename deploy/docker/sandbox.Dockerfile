# syntax=docker/dockerfile:1.7
# The image run_sandbox runs commands in. With sandbox.egress off the container is started with
# --network none. With it on, the container sits on an internal network whose only way out is the
# egress proxy, which refuses private, loopback and link-local destinations.
FROM debian:bookworm-slim

# python3 with the libraries a one-off script reaches for: parsing HTML and XML, tables, dates,
# HTTP. curl and wget go through the proxy. The rest is what a shell one-liner reaches for.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        python3 \
        python3-bs4 \
        python3-lxml \
        python3-requests \
        python3-pandas \
        python3-dateutil \
        python3-yaml \
        python3-tabulate \
        python3-html2text \
        ca-certificates \
        curl \
        wget \
        jq \
        gawk \
        bc \
        coreutils \
        findutils \
        grep \
        sed \
        ripgrep \
        sqlite3 \
        poppler-utils \
        file \
        unzip \
        zip \
        less \
        tree \
    && rm -rf /var/lib/apt/lists/*

# Commands run as nobody against a read-only root, with the workspace mounted at /tmp.
USER 65534:65534
WORKDIR /tmp
ENV HOME=/tmp
ENTRYPOINT []
CMD ["sh"]
