test:
	cargo test

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

docker-shell:
	docker build -t rust-curves .
	docker run -it rust-curves bash

nix-shell:
	nix-shell

benchmark:
	cargo bench
