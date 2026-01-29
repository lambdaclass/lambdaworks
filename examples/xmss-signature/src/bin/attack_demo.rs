//! WOTS+ One-Time Signature Attack Demonstration
//!
//! This binary demonstrates why XMSS is "stateful" and why you must NEVER
//! sign two different messages with the same WOTS+ key (same index).
//!
//! The attack:
//! 1. Alice generates an XMSS key pair
//! 2. Alice signs message M1 with index 0 -> sig1
//! 3. Alice (accidentally) signs message M2 with index 0 -> sig2
//! 4. Attacker observes both signatures
//! 5. Attacker learns positions in each hash chain from both signatures
//! 6. Attacker forges a signature for message M3
//!
//! Run with: cargo run --release -p xmss-signature --bin attack_demo

use sha2::{Digest, Sha256};

/// Reduced parameters for demonstration (smaller than real XMSS but more realistic)
const TOY_N: usize = 16; // Hash output size (bytes) - normally 32
const TOY_W: u32 = 16; // Winternitz parameter - SAME as real XMSS
const TOY_LEN_1: usize = 32; // Message digits (128 bits / 4 bits per digit) - normally 64
const TOY_LEN_2: usize = 3; // Checksum digits (same calculation as real XMSS)
const TOY_LEN: usize = 35; // Total chains - normally 67

fn main() {
    println!("{}", "=".repeat(70));
    println!("WOTS+ ONE-TIME SIGNATURE ATTACK DEMONSTRATION");
    println!("{}", "=".repeat(70));
    println!();
    println!("This demo uses REDUCED parameters to make the attack practical:");
    println!("  Hash output: {} bytes (vs 32 in real XMSS)", TOY_N);
    println!("  Winternitz w: {} (SAME as real XMSS)", TOY_W);
    println!("  Message chains: {} (vs 64 in real XMSS)", TOY_LEN_1);
    println!("  Checksum chains: {} (same as real XMSS)", TOY_LEN_2);
    println!("  Total chains: {} (vs 67 in real XMSS)", TOY_LEN);
    println!();
    println!("The attack is identical - only the hash size is reduced.");
    println!();

    // Step 1: Generate keys
    println!("[Step 1] Alice generates a key pair...");
    let seed = b"demo_secret_seed_1234567890abcdef";
    let public_seed = b"demo_public_seed_abcdef1234567890";

    // Generate WOTS+ secret key (chain starting points)
    let secret_chains: Vec<[u8; TOY_N]> = (0..TOY_LEN)
        .map(|i| {
            let mut h = Sha256::new();
            h.update(seed);
            h.update([i as u8]);
            let result = h.finalize();
            let mut arr = [0u8; TOY_N];
            arr.copy_from_slice(&result[..TOY_N]);
            arr
        })
        .collect();

    // Generate WOTS+ public key (chain endpoints)
    let public_chains: Vec<[u8; TOY_N]> = secret_chains
        .iter()
        .enumerate()
        .map(|(i, start)| chain_hash(start, 0, TOY_W - 1, public_seed, i))
        .collect();

    println!(
        "  Secret key: {} chain starting points",
        secret_chains.len()
    );
    println!("  Public key: {} chain endpoints", public_chains.len());
    println!();

    // Step 2: Sign message M1
    println!("[Step 2] Alice signs message M1 (legitimate)...");
    let m1 = b"Transfer 100 to Bob";
    let m1_digits = message_to_digits(m1);
    let sig1: Vec<[u8; TOY_N]> = secret_chains
        .iter()
        .enumerate()
        .map(|(i, start)| chain_hash(start, 0, m1_digits[i], public_seed, i))
        .collect();

    println!("  M1: \"{}\"", String::from_utf8_lossy(m1));
    println!("  M1 digits: {:?}", m1_digits);
    println!(
        "  Signature: {} chain values at positions {:?}",
        sig1.len(),
        m1_digits
    );

    // Verify
    let verified1 = verify_signature(&sig1, &m1_digits, &public_chains, public_seed);
    println!("  Verified: {}", verified1);
    println!();

    // Step 3: Alice's state rolls back, she signs M2 with SAME key
    println!("[Step 3] DANGER: State rollback! Alice signs M2 with SAME key...");
    let m2 = b"Send 50 to Charlie";
    let m2_digits = message_to_digits(m2);
    let sig2: Vec<[u8; TOY_N]> = secret_chains
        .iter()
        .enumerate()
        .map(|(i, start)| chain_hash(start, 0, m2_digits[i], public_seed, i))
        .collect();

    println!("  M2: \"{}\"", String::from_utf8_lossy(m2));
    println!("  M2 digits: {:?}", m2_digits);

    let verified2 = verify_signature(&sig2, &m2_digits, &public_chains, public_seed);
    println!("  Verified: {}", verified2);
    println!();

    // Step 4: Attacker analyzes both signatures
    println!("[Step 4] Attacker intercepts both signatures...");
    println!();

    // Calculate MIN position per chain (attacker can reach any position >= min)
    let min_positions: Vec<u32> = (0..TOY_LEN)
        .map(|i| std::cmp::min(m1_digits[i], m2_digits[i]))
        .collect();

    println!("  Chain analysis:");
    println!(
        "  {:>6} {:>8} {:>8} {:>8}",
        "Chain", "M1 pos", "M2 pos", "Min"
    );
    for i in 0..TOY_LEN {
        println!(
            "  {:>6} {:>8} {:>8} {:>8}",
            i, m1_digits[i], m2_digits[i], min_positions[i]
        );
    }
    println!();
    println!("  From position p, attacker can chain FORWARD to reach positions >= p");
    println!("  So attacker can forge if target digit >= min(M1 pos, M2 pos) for ALL chains");
    println!();

    // Calculate probability: P(digit >= min_pos) = (W - min_pos) / W
    let prob: f64 = min_positions
        .iter()
        .map(|&m| (TOY_W - m) as f64 / TOY_W as f64)
        .product();
    let expected = if prob > 0.0 {
        (1.0 / prob) as u64
    } else {
        u64::MAX
    };

    println!("  Probability per candidate: {:.4}", prob);
    println!("  Expected attempts: ~{}", expected);
    println!();

    // Step 5: Forge a signature
    println!("[Step 5] Attacker searches for forgeable message...");
    println!();

    let mut forged_msg = None;
    let mut forged_digits = None;

    for i in 0..1_000_000u64 {
        let candidate = format!("Forged tx #{}", i);
        let digits = message_to_digits(candidate.as_bytes());

        // Check if ALL digits are >= min_positions (forgeable)
        // Attacker can chain forward from min_pos to reach any position >= min_pos
        let forgeable = digits
            .iter()
            .enumerate()
            .all(|(j, &d)| d >= min_positions[j]);

        if forgeable {
            println!("  Found forgeable message after {} attempts!", i);
            forged_msg = Some(candidate);
            forged_digits = Some(digits);
            break;
        }

        if i > 0 && i % 100_000 == 0 {
            println!("  ... searched {} candidates", i);
        }
    }

    let (forged_msg, forged_digits) = match (forged_msg, forged_digits) {
        (Some(m), Some(d)) => (m, d),
        _ => {
            println!("  Search exhausted. Try running again with different messages.");
            return;
        }
    };

    println!("  Forged message: \"{}\"", forged_msg);
    println!("  Forged digits: {:?}", forged_digits);
    println!();

    // Step 6: Actually forge the signature
    println!("[Step 6] Constructing forged signature...");
    println!();

    let forged_sig: Vec<[u8; TOY_N]> = (0..TOY_LEN)
        .map(|i| {
            let target = forged_digits[i];

            // Choose source signature with position <= target (can chain forward to reach target)
            // Prefer the one closer to target (fewer steps needed)
            let (source, source_pos) = if m1_digits[i] <= target && m2_digits[i] <= target {
                // Both work, pick the one closer to target
                if m1_digits[i] >= m2_digits[i] {
                    (&sig1[i], m1_digits[i])
                } else {
                    (&sig2[i], m2_digits[i])
                }
            } else if m1_digits[i] <= target {
                (&sig1[i], m1_digits[i])
            } else {
                (&sig2[i], m2_digits[i])
            };

            // Chain forward from source position to target position
            let steps = target - source_pos;
            chain_hash(source, source_pos, steps, public_seed, i)
        })
        .collect();

    println!("  For each chain, attacker either:");
    println!("    - Uses sig1 value and chains forward, OR");
    println!("    - Uses sig2 value and chains forward");
    println!();

    // Show construction
    println!("  Forgery construction:");
    println!(
        "  {:>6} {:>8} {:>8} {:>8} {:>10}",
        "Chain", "Target", "Source", "Steps", "From"
    );
    for i in 0..TOY_LEN {
        let target = forged_digits[i];
        let (source_pos, from) = if m1_digits[i] <= target && m2_digits[i] <= target {
            if m1_digits[i] >= m2_digits[i] {
                (m1_digits[i], "sig1")
            } else {
                (m2_digits[i], "sig2")
            }
        } else if m1_digits[i] <= target {
            (m1_digits[i], "sig1")
        } else {
            (m2_digits[i], "sig2")
        };
        let steps = target - source_pos;
        println!(
            "  {:>6} {:>8} {:>8} {:>8} {:>10}",
            i, target, source_pos, steps, from
        );
    }
    println!();

    // Verify the forged signature
    println!("[Step 7] Verifying forged signature...");
    let verified_forged =
        verify_signature(&forged_sig, &forged_digits, &public_chains, public_seed);

    println!("  Forged signature verifies: {}", verified_forged);
    println!();

    if verified_forged {
        println!("{}", "!".repeat(70));
        println!("  FORGERY SUCCESSFUL!");
        println!("  Alice NEVER signed \"{}\", but it verifies!", forged_msg);
        println!("{}", "!".repeat(70));
    }

    println!();
    println!("{}", "=".repeat(70));
    println!("CONCLUSION");
    println!("{}", "=".repeat(70));
    println!();
    println!("This attack works because WOTS+ signatures reveal hash chain");
    println!("intermediate values. With TWO signatures at the same index:");
    println!();
    println!("  - For each chain, attacker knows values at TWO positions");
    println!("  - Attacker can reach ANY position >= min(pos1, pos2)");
    println!("  - By searching for messages where ALL chains are favorable,");
    println!("    attacker can forge signatures for new messages!");
    println!();
    println!("PROTECTION: Never reuse a WOTS+ key (same index). XMSS handles");
    println!("this by incrementing the index after each signature. But if");
    println!("state is lost (crash, backup restore), reuse becomes possible.");
    println!();
    println!("In production, use monotonic counters, hardware security modules,");
    println!("or other mechanisms to prevent index reuse.");
    println!();
}

/// Hash chain function: iteratively applies hash
fn chain_hash(
    input: &[u8; TOY_N],
    start: u32,
    steps: u32,
    seed: &[u8],
    chain_idx: usize,
) -> [u8; TOY_N] {
    let mut current = *input;
    for step in start..(start + steps) {
        let mut h = Sha256::new();
        h.update(seed);
        h.update([chain_idx as u8]);
        h.update([step as u8]);
        h.update(current);
        let result = h.finalize();
        current.copy_from_slice(&result[..TOY_N]);
    }
    current
}

/// Convert message to WOTS+ digit representation (base-w encoding with checksum)
fn message_to_digits(msg: &[u8]) -> Vec<u32> {
    let mut h = Sha256::new();
    h.update(msg);
    let hash = h.finalize();

    // Extract message digits (4 bits each for w=16)
    let mut digits = Vec::with_capacity(TOY_LEN);

    for i in 0..TOY_LEN_1 {
        let byte_idx = i / 2;
        let digit = if byte_idx < hash.len() {
            if i % 2 == 0 {
                (hash[byte_idx] >> 4) as u32 // High nibble
            } else {
                (hash[byte_idx] & 0x0F) as u32 // Low nibble
            }
        } else {
            0
        };
        digits.push(digit);
    }

    // Compute checksum: sum of (w-1 - digit) for all message digits
    let checksum: u32 = digits.iter().map(|&d| TOY_W - 1 - d).sum();

    // Encode checksum in base w (need TOY_LEN_2 digits)
    let mut cs = checksum;
    for _ in 0..TOY_LEN_2 {
        digits.push(cs % TOY_W);
        cs /= TOY_W;
    }

    digits
}

/// Verify a WOTS+ signature
fn verify_signature(
    sig: &[[u8; TOY_N]],
    msg_digits: &[u32],
    public_key: &[[u8; TOY_N]],
    seed: &[u8],
) -> bool {
    // Complete each chain from signature position to end (w-1)
    for (i, (sig_elem, &digit)) in sig.iter().zip(msg_digits.iter()).enumerate() {
        let steps = TOY_W - 1 - digit;
        let computed_pk = chain_hash(sig_elem, digit, steps, seed, i);

        if computed_pk != public_key[i] {
            return false;
        }
    }
    true
}
