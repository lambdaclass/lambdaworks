//! XMSS (eXtended Merkle Signature Scheme) Implementation
//!
//! This crate provides an educational implementation of XMSS, a hash-based
//! signature scheme defined in [RFC 8391](https://datatracker.ietf.org/doc/html/rfc8391).
//!
//! XMSS provides post-quantum security, meaning it is believed to be secure
//! even against attacks using quantum computers. This makes it a candidate
//! for replacing classical signature schemes like ECDSA in applications
//! requiring long-term security.
//!
//! # Background
//!
//! XMSS is a stateful hash-based signature scheme that:
//! - Uses a Merkle tree of one-time signatures (WOTS+)
//! - Provides 2^h signatures for a tree of height h
//! - Requires careful state management (each key can only be used once)
//!
//! This implementation is inspired by Ethereum's LeanSig proposal for
//! post-quantum consensus signatures.
//!
//! # Usage
//!
//! ```rust
//! use xmss_signature::{Xmss, Sha256Hasher};
//!
//! // Create XMSS instance with SHA-256
//! let xmss = Xmss::new(Sha256Hasher::new());
//!
//! // Generate key pair from random seed
//! let mut seed = [0u8; 96];
//! // In production, use a secure random number generator!
//! // rand::thread_rng().fill_bytes(&mut seed);
//! for i in 0..96 { seed[i] = i as u8; }
//!
//! let (public_key, mut secret_key) = xmss.keygen(&seed);
//!
//! // Sign a message
//! let message = b"Hello, post-quantum world!";
//! let signature = xmss.sign(message, &mut secret_key).expect("signing failed");
//!
//! // Verify the signature
//! assert!(xmss.verify(message, &signature, &public_key));
//! ```
//!
//! # Security Warning
//!
//! This is an **educational implementation** and has not been audited for
//! production use. Key considerations:
//!
//! 1. **Stateful signatures**: XMSS is stateful - each WOTS+ key pair can only
//!    be used once. Reusing a key pair compromises security. The secret key
//!    index must be persisted reliably.
//!
//! 2. **Side channels**: This implementation does not include protections
//!    against timing attacks or other side-channel attacks.
//!
//! 3. **Randomness**: Key generation requires high-quality randomness.
//!
//! # References
//!
//! - [RFC 8391: XMSS](https://datatracker.ietf.org/doc/html/rfc8391)
//! - [Hash-Based Multi-Signatures for Post-Quantum Ethereum](https://eprint.iacr.org/2025/055)
//! - [Lean Consensus Roadmap](https://leanroadmap.org/)

pub mod address;
pub mod hash;
pub mod ltree;
pub mod params;
pub mod utils;
pub mod wots;
pub mod xmss;
pub mod xmss_tree;

// Re-export main types for convenience
pub use address::{Address, AddressType};
pub use hash::{hash_message, Sha256Hasher, XmssHasher};
pub use params::{XmssParams, H, LEN, LEN_1, LEN_2, N, W};
pub use wots::{chain, WotsPublicKey, WotsSignature};
pub use xmss::{Xmss, XmssError, XmssPublicKey, XmssSecretKey, XmssSignature};
pub use xmss_tree::{AuthPath, XmssTree};

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn random_seed() -> [u8; 96] {
        let mut seed = [0u8; 96];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = ((i * 17 + 42) % 256) as u8;
        }
        seed
    }

    #[test]
    fn test_full_sign_verify_cycle() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = random_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        // Sign multiple messages
        let messages = [
            b"First message".as_slice(),
            b"Second message".as_slice(),
            b"Third message with more content to test longer inputs".as_slice(),
            b"".as_slice(), // Empty message
        ];

        let signatures: Vec<_> = messages
            .iter()
            .map(|msg| xmss.sign(msg, &mut sk).expect("signing failed"))
            .collect();

        // Verify all signatures
        for (msg, sig) in messages.iter().zip(signatures.iter()) {
            assert!(
                xmss.verify(msg, sig, &pk),
                "Signature verification failed for message: {:?}",
                std::str::from_utf8(msg)
            );
        }

        // Verify indices are correct
        for (i, sig) in signatures.iter().enumerate() {
            assert_eq!(sig.idx, i as u32);
        }
    }

    #[test]
    fn test_tampered_signature_fails() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = random_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        let message = b"Secure message";
        let mut signature = xmss.sign(message, &mut sk).expect("signing failed");

        // Tamper with the signature
        signature.randomness[0] ^= 0xFF;

        assert!(
            !xmss.verify(message, &signature, &pk),
            "Tampered signature should not verify"
        );
    }

    #[test]
    fn test_signature_with_different_indices() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = random_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        // Skip some indices
        for _ in 0..10 {
            let _ = xmss.sign(b"skip", &mut sk);
        }

        // Sign at index 10
        let message = b"Important message at index 10";
        let signature = xmss.sign(message, &mut sk).expect("signing failed");

        assert_eq!(signature.idx, 10);
        assert!(xmss.verify(message, &signature, &pk));
    }

    #[test]
    fn test_cross_verification_fails() {
        let xmss = Xmss::new(Sha256Hasher::new());

        // Two different key pairs
        let seed1 = random_seed();
        let mut seed2 = random_seed();
        seed2[0] ^= 0xFF;

        let (pk1, mut sk1) = xmss.keygen(&seed1);
        let (pk2, _) = xmss.keygen(&seed2);

        let message = b"Test message";
        let sig1 = xmss.sign(message, &mut sk1).expect("signing failed");

        // Signature from sk1 should verify with pk1
        assert!(xmss.verify(message, &sig1, &pk1));

        // Signature from sk1 should NOT verify with pk2
        assert!(!xmss.verify(message, &sig1, &pk2));
    }

    #[test]
    fn test_two_different_messages_same_key() {
        // This test demonstrates that XMSS can safely sign multiple
        // different messages with the same key pair, as long as each
        // signature uses a different index (which happens automatically).
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = random_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        // Sign two completely different messages
        let message1 = b"Transfer 100 coins to Alice";
        let message2 = b"Transfer 100 coins to Bob";

        assert_eq!(sk.index(), 0);
        let sig1 = xmss.sign(message1, &mut sk).expect("signing failed");
        assert_eq!(sig1.idx, 0); // First signature uses index 0

        assert_eq!(sk.index(), 1);
        let sig2 = xmss.sign(message2, &mut sk).expect("signing failed");
        assert_eq!(sig2.idx, 1); // Second signature uses index 1

        // Both signatures are valid
        assert!(xmss.verify(message1, &sig1, &pk));
        assert!(xmss.verify(message2, &sig2, &pk));

        // But you CANNOT use sig1 to verify message2 or vice versa
        assert!(!xmss.verify(message2, &sig1, &pk));
        assert!(!xmss.verify(message1, &sig2, &pk));

        // The key has been used twice, so remaining = 1024 - 2 = 1022
        assert_eq!(sk.remaining_signatures(), 1022);
    }

    #[test]
    fn test_signature_not_reusable_for_different_message() {
        // Demonstrates that each signature is bound to its specific message.
        // An attacker cannot take a valid signature and use it for a different message.
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = random_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        let original_message = b"I agree to pay $100";
        let forged_message = b"I agree to pay $1000000";

        let signature = xmss
            .sign(original_message, &mut sk)
            .expect("signing failed");

        // Signature is valid for original message
        assert!(xmss.verify(original_message, &signature, &pk));

        // Signature is NOT valid for forged message
        assert!(
            !xmss.verify(forged_message, &signature, &pk),
            "Signature should not verify for a different message"
        );
    }

    #[test]
    fn test_index_reuse_security_demonstration() {
        // SECURITY DEMONSTRATION: This test shows why XMSS is "stateful"
        // and why you must NEVER reuse an index.
        //
        // In XMSS, each leaf corresponds to a WOTS+ one-time signature.
        // If you sign two different messages with the same index, an attacker
        // can potentially forge signatures.
        //
        // This test demonstrates that if you clone the secret key (simulating
        // a state rollback or backup restore), signing different messages
        // produces signatures that both verify - which is the dangerous scenario.

        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = random_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        // Clone the secret key (simulating dangerous state duplication)
        let mut sk_clone = sk.clone();

        let message1 = b"Message A";
        let message2 = b"Message B";

        // Sign message1 with original key
        let sig1 = xmss.sign(message1, &mut sk).expect("signing failed");

        // Sign message2 with cloned key (DANGEROUS: reuses index 0!)
        let sig2 = xmss.sign(message2, &mut sk_clone).expect("signing failed");

        // Both signatures use the same index
        assert_eq!(sig1.idx, sig2.idx, "Both signatures use index 0");

        // Both signatures verify (this is the security problem!)
        assert!(xmss.verify(message1, &sig1, &pk));
        assert!(xmss.verify(message2, &sig2, &pk));

        // In a real attack scenario, having two valid signatures at the same index
        // for different messages can allow an attacker to forge new signatures.
        // This is why XMSS state management is critical!
    }

    #[test]
    fn test_signature_exhaustion_error_handling() {
        // Test the signature exhaustion error case by manually setting
        // the index to the maximum value.
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = random_seed();

        let (_, mut sk) = xmss.keygen(&seed);

        // Manually set index past the maximum (simulating exhaustion)
        // Note: MAX_IDX for h=10 is 1023, so setting idx > 1023 should fail
        sk.idx = 1024;

        let result = xmss.sign(b"This should fail", &mut sk);
        assert!(result.is_err(), "Should fail when signatures are exhausted");
        assert_eq!(result.unwrap_err(), XmssError::SignaturesExhausted);
    }

    #[test]
    fn test_remaining_signatures_tracking() {
        // Test that remaining_signatures correctly tracks usage
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = random_seed();

        let (_, mut sk) = xmss.keygen(&seed);

        assert_eq!(sk.remaining_signatures(), 1024);

        // Sign a few messages
        for i in 0..5 {
            let msg = format!("Message {}", i);
            xmss.sign(msg.as_bytes(), &mut sk).expect("signing failed");
        }

        assert_eq!(sk.remaining_signatures(), 1019);
        assert_eq!(sk.index(), 5);
    }

    #[test]
    fn test_remaining_signatures_exhausted() {
        // Test that remaining_signatures returns 0 when index exceeds MAX_IDX
        // (avoids integer underflow)
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = random_seed();

        let (_, mut sk) = xmss.keygen(&seed);

        // Set index past maximum (simulating exhaustion after last signature)
        sk.idx = 1024;
        assert_eq!(sk.remaining_signatures(), 0);

        // Even larger values should still return 0
        sk.idx = u32::MAX;
        assert_eq!(sk.remaining_signatures(), 0);
    }

    #[test]
    fn test_e2e_realistic_workflow() {
        // End-to-end test simulating a realistic signing workflow
        let xmss = Xmss::new(Sha256Hasher::new());

        // Step 1: Generate keys (in reality, seed should come from secure RNG)
        let mut seed = [0u8; 96];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(7).wrapping_add(13);
        }
        let (public_key, mut secret_key) = xmss.keygen(&seed);

        // Step 2: Serialize public key for distribution
        let pk_bytes = public_key.to_bytes();
        assert_eq!(pk_bytes.len(), 64);

        // Step 3: Deserialize public key (e.g., on verifier's side)
        let recovered_pk = XmssPublicKey::from_bytes(&pk_bytes);
        assert_eq!(public_key.root, recovered_pk.root);

        // Step 4: Sign several transactions
        let transactions = [
            b"tx1: Alice -> Bob: 50 ETH".as_slice(),
            b"tx2: Bob -> Charlie: 25 ETH".as_slice(),
            b"tx3: Charlie -> Alice: 10 ETH".as_slice(),
        ];

        let mut signatures = Vec::new();
        for tx in &transactions {
            let sig = xmss.sign(tx, &mut secret_key).expect("signing failed");
            signatures.push(sig);
        }

        // Step 5: Verify all transactions
        for (tx, sig) in transactions.iter().zip(signatures.iter()) {
            assert!(
                xmss.verify(tx, sig, &recovered_pk),
                "Transaction verification failed"
            );
        }

        // Step 6: Check remaining capacity
        assert_eq!(secret_key.remaining_signatures(), 1024 - 3);

        // Step 7: Demonstrate that old signatures remain valid
        // (signatures don't expire, they're permanently valid)
        assert!(xmss.verify(transactions[0], &signatures[0], &public_key));
    }
}
