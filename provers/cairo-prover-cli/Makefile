.PHONY: test coverage clippy clean

build: 
	cargo build --release

prove: build
	cargo run --release prove $(PROGRAM_PATH) $(PROOF_PATH)

verify: build
	cargo run --release verify $(PROOF_PATH)

run_all: build
	cargo run --release prove_and_verify $(PROGRAM_PATH)
	
clippy:
	cargo clippy --workspace --all-targets -- -D warnings

docker_build_cairo_compiler:
	docker build -f cairo_compile.Dockerfile -t cairo .	
	
docker_compile_cairo:
	docker run -v $(ROOT_DIR):/pwd cairo --proof_mode /pwd/$(PROGRAM) > $(OUTPUT)
