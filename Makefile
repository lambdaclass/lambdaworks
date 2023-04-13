.PHONY: test clippy docker-shell nix-shell benchmarks benchmark docs py.develop

py.develop:
	maturin develop -m py/Cargo.toml

test: py.develop
	cargo test
	python -m unittest py/tests/lamdaworks_py_test.py

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

docs:
	cd docs && mdbook serve --open
