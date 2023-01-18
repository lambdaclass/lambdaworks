FROM rust:1.66

WORKDIR /usr/src/elliptic-curves
COPY . .

CMD cargo test
