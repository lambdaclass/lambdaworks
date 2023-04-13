.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs py.develop

py.develop:
	python3 -m venv .venv
	. .venv/bin/activate
	maturin develop -m py/Cargo.toml

test: py.develop
	cargo test
	python3 -m unittest py/tests/lamdaworks_py_test.py

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
