.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs clean traces

# Proof mode consumes too much memory with cairo-lang to execute
# two instances at the same time in the CI without getting killed
.NOTPARALLEL: $(TEST_TRACES) $(TEST_MEMORIES)

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


NON_PROOF_DIR=proving_system/stark/cairo_programs/non_proof
NON_PROOFS:=$(wildcard $(NON_PROOF_DIR)/*.cairo)
NON_PROOF_TRACES:=$(patsubst $(NON_PROOF_DIR)/%.cairo, $(NON_PROOF_DIR)/%.trace, $(NON_PROOFS))
NON_PROOF_MEMORIES:=$(patsubst $(NON_PROOF_DIR)/%.cairo, $(NON_PROOF_DIR)/%.memory, $(NON_PROOFS))

$(NON_PROOF_DIR)/%.json: $(NON_PROOF_DIR)/%.cairo
	cairo-compile --cairo_path="$(NON_PROOF_DIR)" $< --output $@

$(NON_PROOF_DIR)/%.trace $(NON_PROOF_DIR)/%.memory: $(NON_PROOF_DIR)/%.json
	cairo-run --layout plain --program $< --trace_file $@ --memory_file $(@D)/$(*F).memory


TEST_DIR=proving_system/stark/cairo_programs
TEST_FILES:=$(wildcard $(TEST_DIR)/*.cairo)
COMPILED_TESTS:=$(patsubst $(TEST_DIR)/%.cairo, $(TEST_DIR)/%.json, $(TEST_FILES))
TEST_TRACES:=$(patsubst $(TEST_DIR)/%.cairo, $(TEST_DIR)/%.trace, $(TEST_FILES))
TEST_MEMORIES:=$(patsubst $(TEST_DIR)/%.cairo, $(TEST_DIR)/%.memory, $(TEST_FILES))

$(TEST_DIR)/%.json: $(TEST_DIR)/%.cairo
	cairo-compile --cairo_path="$(TEST_DIR)" $< --output $@ --proof_mode

$(TEST_DIR)/%.trace $(TEST_DIR)/%.memory: $(TEST_DIR)/%.json
	cairo-run --layout plain --proof_mode --program $< --trace_file $@ --memory_file $(@D)/$(*F).memory


traces: $(NON_PROOF_TRACES) $(NON_PROOF_MEMORIES) $(TEST_TRACES) $(TEST_MEMORIES)

benchmarks: traces
	cargo criterion --workspace

# BENCHMARK should be one of the [[bench]] names in Cargo.toml
# Benchmark groups are filtered by name, according to FILTER
# Example: make benchmark BENCH=criterion_field FILTER=CAIRO/fibonacci/50_b4_q64
benchmark: traces
	cargo criterion --bench ${BENCH} -- ${FILTER}

clean:
	rm -f $(TEST_DIR)/*.{json,trace,memory}
	rm -f $(NON_PROOF_DIR)/*.{json,trace,memory}

# Benchmark groups are filtered by name, according to FILTER
# Example: make flamegraph BENCH=criterion_field FILTER=CAIRO/fibonacci/50_b4_q64
flamegraph: traces
	CARGO_PROFILE_BENCH_DEBUG=true cargo flamegraph --root --bench ${BENCH} -- --bench ${FILTER}

METALPATH = gpu/src/metal/shaders
build-metal:
	xcrun -sdk macosx metal $(METALPATH)/all.metal -o $(METALPATH)/lib.metallib

CUDAPATH = gpu/src/cuda/shaders
build-cuda:
	nvcc -ptx $(CUDAPATH)/fft.cu -o $(CUDAPATH)/fft.ptx

docs:
	cd docs && mdbook serve --open
