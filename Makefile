.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs build-cuda build-metal

test:
	cargo test

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

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

METAL_DIR = math/src/gpu/metal/shaders
build-metal:
	xcrun -sdk macosx metal $(METAL_DIR)/all.metal -o $(METAL_DIR)/lib.metallib

CUDA_DIR = math/src/gpu/cuda/shaders
CUDA_FILES:=$(wildcard $(CUDA_DIR)/**/*.cu)
CUDA_COMPILED:=$(patsubst $(CUDA_DIR)/%.cu, $(CUDA_DIR)/%.ptx, $(CUDA_FILES))

$(CUDA_DIR)/%.ptx: $(CUDA_DIR)/%.cu
	nvcc -ptx $< -o $@

# This part compiles all .cu files in $(CUDA_DIR)
build-cuda: $(CUDA_COMPILED)

CUDAPATH = math/src/gpu/cuda/shaders
build-cuda:
	nvcc -ptx $(CUDAPATH)/fft.cu -o $(CUDAPATH)/fft.ptx
	nvcc -ptx $(CUDAPATH)/fields/stark256.cu -o $(CUDAPATH)/fields/stark256.ptx

docs:
	cd docs && mdbook serve --open
