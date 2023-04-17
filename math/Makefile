PORT?=/dev/cu.SLAB_USBtoUART*
PROJECT_NAME=lambdaworks-math

.PHONY: default compile_esp upload log clean

default: clean compile_esp upload

compile_esp: ## Compile rust
	cargo clean
	IDF_PYTHON_ENV_PATH=/opt/homebrew/bin/python3 cargo +nightly build --features esp --release --target riscv32imc-esp-espidf -Z build-std=std,panic_abort

upload: ## Upload the firmware to the board
	espflash $(PORT) target/riscv32imc-esp-espidf/target/$(PROJECT_NAME)

log: ## Start the logs
	espmonitor $(PORT)

clean: ## Remove all build files
	cargo clean
