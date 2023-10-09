.PHONY: test coverage clippy clean

build: 
	cargo build --release

prove: build
	cargo run --release prove $(PROGRAM_PATH) $(PROOF_PATH)

verify: build
	cargo run --release verify $(PROOF_PATH)

run_all: build
	cargo run --release prove-and-verify $(PROGRAM_PATH)
	
clippy:
	cargo clippy --workspace --all-targets -- -D warnings
