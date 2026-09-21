# syntax=docker/dockerfile:1.7
# Egress proxy for the sandbox network: ports 80 and 443 to public addresses only.
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends squid-openssl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/sandbox/squid.conf /etc/squid/squid.conf

USER proxy
EXPOSE 3128
ENTRYPOINT ["squid", "-N", "-f", "/etc/squid/squid.conf"]
