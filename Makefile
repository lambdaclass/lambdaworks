.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs build-cuda build-metal clippy-metal test-metal coverage clean

FUZZ_DIR = fuzz/no_gpu_fuzz

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

CAIRO0_PROGRAMS_DIR=provers/cairo/cairo_programs/cairo0
CAIRO0_PROGRAMS:=$(wildcard $(CAIRO0_PROGRAMS_DIR)/*.cairo)
COMPILED_CAIRO0_PROGRAMS:=$(patsubst $(CAIRO0_PROGRAMS_DIR)/%.cairo, $(CAIRO0_PROGRAMS_DIR)/%.json, $(CAIRO0_PROGRAMS))

# Rule to compile Cairo programs for testing purposes.
# If the `cairo-lang` toolchain is installed, programs will be compiled with it.
# Otherwise, the cairo_compile docker image will be used
# When using the docker version, be sure to build the image using `make docker_build_cairo_compiler`.
$(CAIRO0_PROGRAMS_DIR)/%.json: $(CAIRO0_PROGRAMS_DIR)/%.cairo
	@echo "Compiling Cairo program..."
	@cairo-compile --cairo_path="$(CAIRO0_PROGRAMS_DIR)" $< --output $@ 2> /dev/null --proof_mode || \
	docker run --rm -v $(ROOT_DIR)/$(CAIRO0_PROGRAMS_DIR):/pwd/$(CAIRO0_PROGRAMS_DIR) cairo --proof_mode /pwd/$< > $@

test: $(COMPILED_CAIRO0_PROGRAMS)
	cargo test

clippy:
	cargo clippy --workspace -- -D warnings
	cargo clippy --tests

clippy-metal:
	cargo clippy --workspace -F metal -- -D warnings

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

coverage: $(COMPILED_CAIRO0_PROGRAMS)
	cargo llvm-cov nextest --lcov --output-path lcov.info
	
METAL_DIR = math/src/gpu/metal
build-metal:
	xcrun -sdk macosx metal $(METAL_DIR)/all.metal -o $(METAL_DIR)/lib.metallib

clippy-metal:
	cargo clippy --workspace --all-targets -F metal -- -D warnings

test-metal: $(COMPILED_CAIRO0_PROGRAMS)
	cargo test -F metal

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
		
run-metal-fuzzer:
		cd fuzz/metal_fuzz
		cargo +nightly fuzz run --fuzz-dir  $(FUZZ_DIR) fft_diff

run-cuda-fuzzer:
		cd fuzz/cuda_fuzz
		cargo hfuzz run $(CUDAFUZZER)

