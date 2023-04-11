.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs

test:
	cargo test

clippy:
	cargo clippy --all-targets -- -D warnings

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

METALPATH = gpu/src/metal/shaders
build-metal:
	xcrun -sdk macosx metal $(METALPATH)/all.metal -o $(METALPATH)/lib.metallib

docs:
	cd docs && mdbook serve --open

compile-cuda-fatbin:
	nvcc -fatbin -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 --x cu gpu/src/cuda/fft.cu -o gpu/src/cuda/fft.fatbin
