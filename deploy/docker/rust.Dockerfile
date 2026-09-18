# syntax=docker/dockerfile:1.7
# One image for both Rust apps; compose entrypoint selects the binary.
FROM rust:1.95-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release -p engine -p discord

# Not distroless: run_sandbox shells out to the container runtime client, which needs a binary
# and a libc. The client talks to the sandboxd daemon, never to the host runtime.
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/engine /engine
COPY --from=builder /app/target/release/discord /discord
# Settings layer at /, overridden by environment.
COPY sparky.toml /sparky.toml
EXPOSE 8080
ENTRYPOINT ["/engine"]
