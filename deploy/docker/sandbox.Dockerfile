# syntax=docker/dockerfile:1.7
# The image run_sandbox runs commands in. It holds no network tooling: the container is started
# with --network none, and the agent is told to use a search tool to fetch anything.
FROM debian:bookworm-slim

# python3 covers dates, arithmetic, parsing and reshaping through its standard library. jq covers
# the JSON a tool result arrives as. The rest is what a shell one-liner reaches for.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        python3 \
        jq \
        gawk \
        bc \
        coreutils \
        findutils \
        grep \
        sed \
    && rm -rf /var/lib/apt/lists/*

# Commands run as nobody against a read-only root, with the workspace mounted at /tmp.
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT []
CMD ["sh"]
