test:
	cargo test

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

docker-shell:
	docker build -t rust-curves .
	docker run -it rust-curves bash

nix-shell:
	nix-shell

benchmarks:
	cargo criterion --bench all_benchmarks

# BENCHMARK should be one of the [[bench]] names in Cargo.toml
benchmark:
	cargo criterion --bench ${BENCH}

.PHONY: clean
clean:
	cargo clean

.PHONY: build.lib.py
## Compiles the .dylib and generates a python wrapper
build.lib.py:
	cargo run --release --bin lambdaworks-stark generate proving-system/stark/src/lambdaworks_stark.udl --language python --out-dir target/python/ --no-format
	cp target/release/deps/liblambdaworks_stark.dylib target/python/liblambdaworks.dylib
