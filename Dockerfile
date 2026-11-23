FROM rust:1.100-slim-bookworm AS builder

WORKDIR /usr/src/serenade

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        llvm \
        clang;

COPY ./src src
COPY ./Cargo.* ./

RUN rustup component add rustfmt
RUN cargo build --locked --release --bin serving

FROM debian:bookworm-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/src/serenade/target/release/serving /app

EXPOSE 8080
ENTRYPOINT [ "/app/serving" ]
