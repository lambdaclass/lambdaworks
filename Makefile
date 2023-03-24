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

build-metal:
	xcrun -sdk macosx metal \
	gpu/src/metal/shaders/fft.metal \
	gpu/src/metal/shaders/twiddles.metal \
	-o gpu/src/metal/shaders/lib.metallib

