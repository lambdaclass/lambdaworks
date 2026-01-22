//! Hash functions for XMSS
//!
//! This module defines the hash function trait required by XMSS and provides
//! a SHA-256 based implementation following RFC 8391.
//!
//! XMSS requires four hash function instances:
//! - F: Chain function for WOTS+ (n bytes -> n bytes)
//! - H: Tree hashing function (2n bytes -> n bytes)
//! - H_msg: Message hash with randomization
//! - PRF: Pseudorandom function for key generation

use crate::address::Address;
use crate::params::N;
use sha2::{Digest, Sha256};

/// Hash function trait for XMSS
///
/// Implementations must provide domain-separated hash functions
/// as specified in RFC 8391.
pub trait XmssHasher: Clone + Default {
    /// F function: chain hash used in WOTS+
    ///
    /// Computes: F(KEY || M) where KEY = PRF(SEED, ADRS)
    fn f(&self, input: &[u8; N], public_seed: &[u8; N], address: &Address) -> [u8; N];

    /// H function: tree hash for internal nodes
    ///
    /// Computes: H(KEY || M) where M is concatenation of two child nodes
    fn h(&self, left: &[u8; N], right: &[u8; N], public_seed: &[u8; N], address: &Address)
        -> [u8; N];

    /// H_msg function: randomized message hashing
    ///
    /// Computes: H_msg(r || root || idx || M)
    fn h_msg(
        &self,
        randomness: &[u8; N],
        root: &[u8; N],
        index: u64,
        message: &[u8],
    ) -> [u8; N];

    /// PRF function: pseudorandom function for key derivation
    ///
    /// Computes: PRF(KEY, M) for deriving keys from seeds
    fn prf(&self, key: &[u8; N], input: &[u8; 32]) -> [u8; N];

    /// PRF_keygen: specialized PRF for secret key generation
    fn prf_keygen(&self, secret_seed: &[u8; N], address: &Address) -> [u8; N];
}

/// SHA-256 based hasher following RFC 8391
///
/// This implementation uses SHA-256 as the underlying hash function
/// with proper padding and domain separation.
#[derive(Clone, Default)]
pub struct Sha256Hasher;

impl Sha256Hasher {
    /// Create a new SHA-256 hasher
    pub fn new() -> Self {
        Self
    }

    /// Compute PRF using SHA-256
    /// PRF(KEY, M) = SHA-256(toByte(3, 32) || KEY || M)
    fn prf_internal(&self, key: &[u8; N], input: &[u8]) -> [u8; N] {
        let mut hasher = Sha256::new();

        // Domain separator: toByte(3, 32) - padding to 32 bytes with value 3
        let mut padding = [0u8; 32];
        padding[31] = 3;
        hasher.update(&padding);

        hasher.update(key);
        hasher.update(input);

        let result = hasher.finalize();
        let mut output = [0u8; N];
        output.copy_from_slice(&result);
        output
    }
}

impl XmssHasher for Sha256Hasher {
    fn f(&self, input: &[u8; N], public_seed: &[u8; N], address: &Address) -> [u8; N] {
        // F(KEY || M) where KEY = PRF(SEED, ADRS)
        let key = self.prf_internal(public_seed, address.as_bytes());

        let mut hasher = Sha256::new();

        // Domain separator: toByte(0, 32)
        let mut padding = [0u8; 32];
        padding[31] = 0;
        hasher.update(&padding);

        hasher.update(&key);
        hasher.update(input);

        let result = hasher.finalize();
        let mut output = [0u8; N];
        output.copy_from_slice(&result);
        output
    }

    fn h(
        &self,
        left: &[u8; N],
        right: &[u8; N],
        public_seed: &[u8; N],
        address: &Address,
    ) -> [u8; N] {
        // H(KEY || left || right) where KEY = PRF(SEED, ADRS)
        let key = self.prf_internal(public_seed, address.as_bytes());

        let mut hasher = Sha256::new();

        // Domain separator: toByte(1, 32)
        let mut padding = [0u8; 32];
        padding[31] = 1;
        hasher.update(&padding);

        hasher.update(&key);
        hasher.update(left);
        hasher.update(right);

        let result = hasher.finalize();
        let mut output = [0u8; N];
        output.copy_from_slice(&result);
        output
    }

    fn h_msg(
        &self,
        randomness: &[u8; N],
        root: &[u8; N],
        index: u64,
        message: &[u8],
    ) -> [u8; N] {
        let mut hasher = Sha256::new();

        // Domain separator: toByte(2, 32)
        let mut padding = [0u8; 32];
        padding[31] = 2;
        hasher.update(&padding);

        hasher.update(randomness);
        hasher.update(root);

        // Index as 32 bytes (big-endian, zero-padded)
        let mut idx_bytes = [0u8; 32];
        idx_bytes[24..32].copy_from_slice(&index.to_be_bytes());
        hasher.update(&idx_bytes);

        hasher.update(message);

        let result = hasher.finalize();
        let mut output = [0u8; N];
        output.copy_from_slice(&result);
        output
    }

    fn prf(&self, key: &[u8; N], input: &[u8; 32]) -> [u8; N] {
        self.prf_internal(key, input)
    }

    fn prf_keygen(&self, secret_seed: &[u8; N], address: &Address) -> [u8; N] {
        self.prf_internal(secret_seed, address.as_bytes())
    }
}

/// Simple hash of arbitrary data (utility function)
pub fn hash_message(data: &[u8]) -> [u8; N] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; N];
    output.copy_from_slice(&result);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_hasher_f_deterministic() {
        let hasher = Sha256Hasher::new();
        let input = [0xABu8; N];
        let seed = [0x42u8; N];
        let address = Address::new();

        let result1 = hasher.f(&input, &seed, &address);
        let result2 = hasher.f(&input, &seed, &address);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_sha256_hasher_f_different_inputs() {
        let hasher = Sha256Hasher::new();
        let input1 = [0x00u8; N];
        let input2 = [0x01u8; N];
        let seed = [0x42u8; N];
        let address = Address::new();

        let result1 = hasher.f(&input1, &seed, &address);
        let result2 = hasher.f(&input2, &seed, &address);

        assert_ne!(result1, result2);
    }

    #[test]
    fn test_sha256_hasher_h_deterministic() {
        let hasher = Sha256Hasher::new();
        let left = [0xAAu8; N];
        let right = [0xBBu8; N];
        let seed = [0x42u8; N];
        let address = Address::new();

        let result1 = hasher.h(&left, &right, &seed, &address);
        let result2 = hasher.h(&left, &right, &seed, &address);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_sha256_hasher_h_order_matters() {
        let hasher = Sha256Hasher::new();
        let left = [0xAAu8; N];
        let right = [0xBBu8; N];
        let seed = [0x42u8; N];
        let address = Address::new();

        let result1 = hasher.h(&left, &right, &seed, &address);
        let result2 = hasher.h(&right, &left, &seed, &address);

        assert_ne!(result1, result2);
    }

    #[test]
    fn test_sha256_hasher_h_msg() {
        let hasher = Sha256Hasher::new();
        let randomness = [0x11u8; N];
        let root = [0x22u8; N];
        let message = b"test message";

        let result1 = hasher.h_msg(&randomness, &root, 0, message);
        let result2 = hasher.h_msg(&randomness, &root, 0, message);

        assert_eq!(result1, result2);

        // Different index should give different result
        let result3 = hasher.h_msg(&randomness, &root, 1, message);
        assert_ne!(result1, result3);
    }

    #[test]
    fn test_sha256_hasher_prf() {
        let hasher = Sha256Hasher::new();
        let key = [0x42u8; N];
        let input = [0xABu8; 32];

        let result1 = hasher.prf(&key, &input);
        let result2 = hasher.prf(&key, &input);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_hash_message() {
        let msg1 = b"hello";
        let msg2 = b"world";

        let hash1 = hash_message(msg1);
        let hash2 = hash_message(msg2);

        assert_ne!(hash1, hash2);
        assert_eq!(hash1.len(), N);
    }
}
