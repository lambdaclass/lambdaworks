//! Hash-to-curve implementations following RFC 9380.
//!
//! Currently supports:
//! - BLS12-381 G1: `hash_to_g1` (suite `BLS12381G1_XMD:SHA-256_SSWU_RO_`)
//! - BLS12-381 G2: `hash_to_g2` (suite `BLS12381G2_XMD:SHA-256_SSWU_RO_`)

pub mod bls12_381;
