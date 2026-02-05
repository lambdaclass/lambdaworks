# BLS Signatures Example

This example demonstrates BLS (Boneh-Lynn-Shacham) signatures using the BLS12-381 curve from lambdaworks.

## Features

- **Basic signing and verification**: Sign messages and verify signatures
- **Signature aggregation**: Combine multiple signatures into one (constant size!)
- **Public key aggregation**: Combine multiple public keys
- **Batch verification**: Efficiently verify multiple signatures

## Usage

```rust
use bls_signature::{SecretKey, aggregate_signatures, verify_aggregated};

// Generate keys
let sk1 = SecretKey::from_seed(b"signer 1").unwrap();
let sk2 = SecretKey::from_seed(b"signer 2").unwrap();
let pk1 = sk1.public_key();
let pk2 = sk2.public_key();

// Sign a message
let message = b"Hello, BLS!";
let sig1 = sk1.sign(message);
let sig2 = sk2.sign(message);

// Verify individual signatures
assert!(pk1.verify(message, &sig1).is_ok());
assert!(pk2.verify(message, &sig2).is_ok());

// Aggregate signatures (2 signatures -> 1 signature!)
let agg_sig = aggregate_signatures(&[sig1, sig2]).unwrap();

// Verify aggregated signature
assert!(verify_aggregated(message, &agg_sig, &[pk1, pk2]).is_ok());
```

## Why BLS?

BLS signatures have unique properties that make them ideal for:

1. **Ethereum 2.0 consensus**: Validators sign attestations, and thousands of signatures are aggregated into one
2. **Threshold signatures**: Combine partial signatures from multiple parties
3. **Space efficiency**: 96-byte signatures regardless of how many signers

### Size Comparison

| Signers | ECDSA Total | BLS Aggregated |
|---------|-------------|----------------|
| 1       | 64 bytes    | 96 bytes       |
| 10      | 640 bytes   | 96 bytes       |
| 100     | 6,400 bytes | 96 bytes       |
| 10,000  | 640 KB      | 96 bytes       |

## Security Warning

This implementation is for **educational purposes only**. For production use:

- Use audited libraries like `blst` or `bls-signatures`
- Implement proof-of-possession to prevent rogue key attacks
- Follow the IETF hash-to-curve standard (RFC 9380)

## Running Tests

```bash
cargo test -p bls-signature
```
