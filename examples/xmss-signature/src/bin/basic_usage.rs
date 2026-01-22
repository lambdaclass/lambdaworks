//! Basic XMSS Usage Example
//!
//! This binary demonstrates the basic keygen -> sign -> verify flow
//! using the full XMSS implementation.
//!
//! Run with: cargo run --release -p xmss-signature --bin basic_usage

use xmss_signature::{Sha256Hasher, Xmss};

fn main() {
    println!("======================================================================");
    println!("XMSS Basic Usage Example");
    println!("======================================================================");
    println!();

    // Create XMSS instance with SHA-256 hasher
    let xmss = Xmss::new(Sha256Hasher::new());

    println!("[1] Generating XMSS key pair...");
    println!("    Parameters: n=32, w=16, h=10 (1024 signatures)");
    println!("    This takes ~45 seconds as it builds a Merkle tree of 1024 WOTS+ keys");
    println!();

    // Generate seed (in production, use a secure RNG!)
    let mut seed = [0u8; 96];
    for i in 0..96 {
        seed[i] = (i as u8).wrapping_mul(17).wrapping_add(42);
    }

    let (public_key, mut secret_key) = xmss.keygen(&seed);

    println!("    Public key root: {:02x?}...", &public_key.root[..8]);
    println!("    Public seed: {:02x?}...", &public_key.public_seed[..8]);
    println!("    Remaining signatures: {}", secret_key.remaining_signatures());
    println!();

    // Sign first message
    println!("[2] Signing messages...");
    let messages = [
        "Transfer 100 ETH to Alice",
        "Approve smart contract deployment",
        "Vote YES on proposal #42",
    ];

    let mut signatures = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        let sig = xmss.sign(msg.as_bytes(), &mut secret_key).expect("signing failed");
        println!("    Message {}: \"{}\"", i + 1, msg);
        println!("      Signature index: {}", sig.idx);
        println!("      Signature size: {} bytes",
            4 + 32 + sig.wots_sig.sig.len() * 32 + sig.auth_path.path.len() * 32);
        signatures.push(sig);
    }
    println!();
    println!("    Remaining signatures: {}", secret_key.remaining_signatures());
    println!();

    // Verify signatures
    println!("[3] Verifying signatures...");
    for (i, (msg, sig)) in messages.iter().zip(signatures.iter()).enumerate() {
        let valid = xmss.verify(msg.as_bytes(), sig, &public_key);
        println!("    Message {}: verified = {}", i + 1, valid);
    }
    println!();

    // Demonstrate that signatures are message-bound
    println!("[4] Security check: signature is bound to its message...");
    let wrong_msg = "Transfer 1000000 ETH to Eve";
    let valid = xmss.verify(wrong_msg.as_bytes(), &signatures[0], &public_key);
    println!("    Original: \"{}\"", messages[0]);
    println!("    Attempted: \"{}\"", wrong_msg);
    println!("    Signature verifies for wrong message: {}", valid);
    println!();

    // Show public key serialization
    println!("[5] Public key serialization...");
    let pk_bytes = public_key.to_bytes();
    println!("    Serialized size: {} bytes", pk_bytes.len());
    println!("    Format: 32-byte root || 32-byte public_seed");
    println!();

    println!("======================================================================");
    println!("Summary");
    println!("======================================================================");
    println!();
    println!("XMSS provides post-quantum secure signatures with:");
    println!("  - 128-bit security level (with SHA-256, n=32)");
    println!("  - ~2.5 KB signatures (with h=10)");
    println!("  - 64 byte public keys");
    println!("  - 1024 signatures per key pair (with h=10)");
    println!();
    println!("Trade-offs:");
    println!("  - Stateful: must track which indices have been used");
    println!("  - Slow key generation: O(2^h) hash operations");
    println!("  - Large signatures compared to ECDSA (~64 bytes)");
    println!();
}
