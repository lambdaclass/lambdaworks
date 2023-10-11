.PHONY: test coverage clippy clean

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Variable to give compiled Cairo programas a proper name. It extracts the file extension and adds
# the .json extension to it.
COMPILED_PROGRAM=$(basename $(PROGRAM_PATH)).json

build: 
	cargo build --release

docker_build_cairo_compiler:
	docker build -f cairo_compile.Dockerfile -t cairo .

compile:
	cairo-compile --proof_mode $(PROGRAM_PATH) > $(COMPILED_PROGRAM) 2>/dev/null || \
	docker run -v $(ROOT_DIR):/pwd cairo --proof_mode /pwd/$(PROGRAM_PATH) > $(COMPILED_PROGRAM)

prove: build
	cargo run --release prove $(PROGRAM_PATH) $(PROOF_PATH)

verify: build
	cargo run --release verify $(PROOF_PATH)

run_all: build
	cargo run --release prove-and-verify $(PROGRAM_PATH)
	
clippy:
	cargo clippy --workspace --all-targets -- -D warnings
