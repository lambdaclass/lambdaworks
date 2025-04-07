.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs build-cuda coverage clean

FUZZ_DIR = fuzz/no_gpu_fuzz

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

test:
	cargo test

clippy:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo clippy --workspace --all-targets --features wasm -- -D warnings
	cargo clippy --workspace --all-targets --features parallel -- -D warnings
	cargo clippy --tests

clippy-cuda:
	cargo clippy --workspace -F cuda -- -D warnings

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

coverage:
	cargo llvm-cov nextest --lcov --output-path lcov.info

CUDA_DIR = math/src/gpu/cuda/shaders
CUDA_FILES:=$(wildcard $(CUDA_DIR)/**/*.cu)
CUDA_COMPILED:=$(patsubst $(CUDA_DIR)/%.cu, $(CUDA_DIR)/%.ptx, $(CUDA_FILES))
CUDA_HEADERS:=$(wildcard $(CUDA_DIR)/**/*.cuh)

$(CUDA_DIR)/%.ptx: $(CUDA_DIR)/%.cu $(CUDA_HEADERS)
	nvcc -ptx $< -o $@

# This part compiles all .cu files in $(CUDA_DIR)
build-cuda: $(CUDA_COMPILED)

CUDAPATH = math/src/gpu/cuda/shaders
build-cuda:
	nvcc -ptx $(CUDAPATH)/field/stark256.cu -o $(CUDAPATH)/field/stark256.ptx

docs:
	cd docs && mdbook serve --open

run-fuzzer:
		cargo +nightly fuzz run --fuzz-dir $(FUZZ_DIR) $(FUZZER)

proof-deserializer-fuzzer:
		cargo +nightly fuzz run --fuzz-dir $(FUZZ_DIR)  deserialize_stark_proof

run-cuda-fuzzer:
		cd fuzz/cuda_fuzz
		cargo hfuzz run $(CUDAFUZZER)
