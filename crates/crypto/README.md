# lambdaworks-crypto [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/lambdaworks-crypto.svg
[crates.io]: https://crates.io/crates/lambdaworks-crypto

## Usage

Add this to your `Cargo.toml`
```toml
[dependencies]
lambdaworks-crypto = "0.12.0"
```

## Structure

This crate contains different cryptographic primitives needed for proof systems. The main elements are:
- [Merkle trees](./src/merkle_tree/)
- [Hash functions](./src/hash/)
- [Fiat Shamir transformation](./src/fiat_shamir/)
- [Polynomial commitment schemes](./src/commitments/)

For examples on:
- How do Merkle trees work, refer to [Merkle CLI](../../examples/merkle-tree-cli/README.md)
- Hash functions, refer to [Hash functions' readme](./src/hash/README.md)
- Fiat-Shamir heuristic, refer to [Fiat-Shamir's readme](./src/fiat_shamir/README.md)
- Polynomial commitment schemes, refer to [PCS's readme](./src/commitments/README.md)

