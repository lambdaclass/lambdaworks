test:
	cargo test

clippy:
	cargo clippy --all-targets --all-features -- -W clippy::as_conversions -D warnings
	# cargo clippy --all-targets --all-features -- -W clippy::clippy::as_conversions -- -D warnings

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


	# u64::try_from(u8::MAX).unwrap()
	# u64::try_from(ORDER - 1).unwrap()
	# let twiddles = (0..(u64::try_from(coeffs.len()).unwrap())).map(|i| root.pow(i)).collect::<Vec<FE>>();
