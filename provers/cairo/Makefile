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

prove:
	cargo run --quiet --release prove $(PROGRAM_PATH) $(PROOF_PATH)

verify:
	cargo run --quiet --release verify $(PROOF_PATH)

run_all:
	cargo run --quiet --release prove-and-verify $(PROGRAM_PATH)

compile_and_prove:
	@cairo-compile --proof_mode $(PROGRAM_PATH) > $(COMPILED_PROGRAM) 2>/dev/null || \
	docker run -v $(ROOT_DIR):/pwd cairo --proof_mode /pwd/$(PROGRAM_PATH) > $(COMPILED_PROGRAM)
	@echo "Compiling done \n"
	@cargo run --quiet --release prove $(COMPILED_PROGRAM) $(PROOF_PATH)
	@rm $(COMPILED_PROGRAM)

compile_and_run_all:
	@cairo-compile --proof_mode $(PROGRAM_PATH) > $(COMPILED_PROGRAM) 2>/dev/null || \
	docker run -v $(ROOT_DIR):/pwd cairo --proof_mode /pwd/$(PROGRAM_PATH) > $(COMPILED_PROGRAM)
	@echo "Compiling done \n"
	@cargo run --quiet --release prove-and-verify $(COMPILED_PROGRAM) $(PROOF_PATH)
	@rm $(COMPILED_PROGRAM)

clippy:
	cargo clippy --workspace --all-targets -- -D warnings
