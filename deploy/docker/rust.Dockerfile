# syntax=docker/dockerfile:1.7
# One image for both Rust apps; select the binary with `command:`.
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

FROM gcr.io/distroless/cc-debian12
COPY --from=builder /app/target/release/engine /engine
COPY --from=builder /app/target/release/discord /discord
EXPOSE 8080
ENTRYPOINT ["/engine"]
