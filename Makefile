.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs

deps:
	pyenv install -s 3.9.15
	PYENV_VERSION=3.9.15 python -m venv venv
	. venv/bin/activate ; \
	pip install cairo-lang==0.11.0

test:
	cargo test

clippy:
	cargo clippy --all-targets -- -D warnings

docker-shell:
	docker build -t rust-curves .
	docker run -it rust-curves bash

nix-shell:
	nix-shell

TEST_DIR=proving_system/stark/cairo_programs
TEST_FILES:=$(wildcard $(TEST_DIR)/*.cairo)
COMPILED_TESTS:=$(patsubst $(TEST_DIR)/%.cairo, $(TEST_DIR)/%.json, $(TEST_FILES))
TEST_TRACES:=$(patsubst $(TEST_DIR)/%.cairo, $(TEST_DIR)/%.trace, $(TEST_FILES))

$(TEST_DIR)/%.json: $(TEST_DIR)/%.cairo
	cairo-compile --cairo_path="$(TEST_DIR)" $< --output $@ --proof_mode

benchmarks: $(COMPILED_TESTS)
	cargo criterion --workspace

# BENCHMARK should be one of the [[bench]] names in Cargo.toml
benchmark: $(COMPILED_TESTS)
	cargo criterion --bench ${BENCH}

flamegraph_stark: $(COMPILED_TESTS)
	CARGO_PROFILE_BENCH_DEBUG=true cargo flamegraph --root --bench criterion_stark -- --bench

METALPATH = gpu/src/metal/shaders
build-metal:
	xcrun -sdk macosx metal $(METALPATH)/all.metal -o $(METALPATH)/lib.metallib

docs:
	cd docs && mdbook serve --open
