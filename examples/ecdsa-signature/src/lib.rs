//! ECDSA signature scheme for secp256k1 curve.
//!
//! This implementation is NOT constant-time and should only be used for
//! signature verification or in non-production/testing contexts.
//!
//! For production signing, use a constant-time implementation.

mod ecdsa;

pub use ecdsa::*;
