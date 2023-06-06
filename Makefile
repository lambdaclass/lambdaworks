.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs build-cuda build-metal

test:
	cargo test

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

clippy-metal:
	cargo clippy --workspace --all-targets -F metal -- -D warnings

clippy-cuda:
	cargo clippy --workspace --all-targets -F cuda -- -D warnings

docker-shell:
	docker build -t rust-curves .
	docker run -it rust-curves bash

nix-shell:
	nix-shell

benchmarks:
	cargo criterion --workspace

# BENCHMARK should be one of the [[bench]] names in Cargo.toml
benchmark:
	cargo criterion --bench ${BENCH}

flamegraph_stark:
	CARGO_PROFILE_BENCH_DEBUG=true cargo flamegraph --root --bench stark_benchmarks -- --bench

docs:
	cd docs && mdbook serve --open

METAL_DIR = gpu/src/metal/shaders
build-metal:
	xcrun -sdk macosx metal $(METAL_DIR)/all.metal -o $(METAL_DIR)/lib.metallib


CUDA_DIR = gpu/src/cuda/shaders
CUDA_FILES:=$(wildcard $(CUDA_DIR)/*.cu)
CUDA_COMPILED:=$(patsubst $(CUDA_DIR)/%.cu, $(CUDA_DIR)/%.ptx, $(CUDA_FILES))
CUDA_HEADERS:=$(wildcard $(CUDA_DIR)/**/*.cuh)

$(CUDA_DIR)/%.ptx: $(CUDA_DIR)/%.cu $(CUDA_HEADERS)
	nvcc -ptx $< -o $@

# This part compiles all .cu files in $(CUDA_DIR)
build-cuda: $(CUDA_COMPILED)

CUDAPATH = gpu/src/cuda/shaders
build-cuda:
	nvcc -ptx $(CUDAPATH)/fields/stark256.cu -o $(CUDAPATH)/fields/stark256.ptx

docs:
	cd docs && mdbook serve --open
