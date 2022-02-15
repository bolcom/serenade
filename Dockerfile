FROM rust:slim-buster AS builder

WORKDIR /usr/src/serenade

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        llvm \
        clang;

COPY ./src src
COPY ./benches benches
COPY ./Cargo.* ./

RUN rustup component add rustfmt
RUN cargo build --release

FROM debian:bullseye-slim as runtime

WORKDIR /app

COPY --from=builder /usr/src/serenade/target/release/serving /app

EXPOSE 8080
ENTRYPOINT [ "/app/serving" ]
