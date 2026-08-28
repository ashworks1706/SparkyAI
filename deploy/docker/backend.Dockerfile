# syntax=docker/dockerfile:1.7
FROM rust:1.95-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

FROM chef AS planner
COPY apps/backend .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY apps/backend .
RUN cargo build --release -p sparky-app

FROM gcr.io/distroless/cc-debian12
COPY --from=builder /app/target/release/sparky-app /sparky
COPY apps/backend/crates/storage/migrations /migrations
EXPOSE 8080
ENTRYPOINT ["/sparky"]
CMD ["api"]
