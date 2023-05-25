.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs clean traces build-cuda build-metal


deps:
	pyenv install -s 3.9.15
	PYENV_VERSION=3.9.15 python -m venv venv
	. venv/bin/activate ; \
	pip install cairo-lang==0.11.0

test:
	cargo test

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

docker-shell:
	docker build -t rust-curves .
	docker run -it rust-curves bash

nix-shell:
	nix-shell


TEST_DIR=proving_system/stark/cairo_programs
TEST_FILES:=$(wildcard $(TEST_DIR)/*.cairo)
COMPILED_TESTS:=$(patsubst $(TEST_DIR)/%.cairo, $(TEST_DIR)/%.json, $(TEST_FILES))
TEST_TRACES:=$(patsubst $(TEST_DIR)/%.cairo, $(TEST_DIR)/%.trace, $(TEST_FILES))
TEST_MEMORIES:=$(patsubst $(TEST_DIR)/%.cairo, $(TEST_DIR)/%.memory, $(TEST_FILES))

$(TEST_DIR)/%.json: $(TEST_DIR)/%.cairo
	cairo-compile --cairo_path="$(TEST_DIR)" $< --output $@

$(TEST_DIR)/%.trace $(TEST_DIR)/%.memory: $(TEST_DIR)/%.json
	cairo-run --layout plain --program $< --trace_file $@ --memory_file $(@D)/$(*F).memory


traces: $(TEST_TRACES) $(TEST_MEMORIES)

benchmarks: traces
	cargo criterion --workspace

# BENCHMARK should be one of the [[bench]] names in Cargo.toml
# Benchmark groups are filtered by name, according to FILTER
# Example: make benchmark BENCH=criterion_field FILTER=CAIRO/fibonacci/50_b4_q64
benchmark: traces
	cargo criterion --bench ${BENCH} -- ${FILTER}

clean:
	rm -f $(TEST_DIR)/*.{json,trace,memory}

# Benchmark groups are filtered by name, according to FILTER
# Example: make flamegraph BENCH=criterion_field FILTER=CAIRO/fibonacci/50_b4_q64
flamegraph: traces
	CARGO_PROFILE_BENCH_DEBUG=true cargo flamegraph --root --bench ${BENCH} -- --bench ${FILTER}

METAL_DIR = gpu/src/metal/shaders
build-metal:
	xcrun -sdk macosx metal $(METAL_DIR)/all.metal -o $(METAL_DIR)/lib.metallib

CUDA_DIR = gpu/src/cuda/shaders
CUDA_FILES:=$(wildcard $(CUDA_DIR)/**/*.cu)
CUDA_COMPILED:=$(patsubst $(CUDA_DIR)/%.cu, $(CUDA_DIR)/%.ptx, $(CUDA_FILES))

$(CUDA_DIR)/%.ptx: $(CUDA_DIR)/%.cu
	nvcc -ptx $< -o $@

# This part compiles all .cu files in $(CUDA_DIR)
build-cuda: $(CUDA_COMPILED)

CUDAPATH = gpu/src/cuda/shaders
build-cuda:
	nvcc -ptx $(CUDAPATH)/fft.cu -o $(CUDAPATH)/fft.ptx

docs:
	cd docs && mdbook serve --open
