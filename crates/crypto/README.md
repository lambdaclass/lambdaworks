# lambdaworks-crypto [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/lambdaworks-crypto.svg
[crates.io]: https://crates.io/crates/lambdaworks-crypto

## Usage

Add this to your `Cargo.toml`
```toml
[dependencies]
lambdaworks-crypto = "0.11.0"
```

## Structure

This crate contains different cryptographic primitives needed for proof systems. The main elements are:
- [Merkle trees](./src/merkle_tree/)
- [Hash functions](./src/hash/)
- [Fiat Shamir transformation](./src/fiat_shamir/)
- [Polynomial commitment schemes](./src/commitments/)
