//! FROST 2-of-2 Threshold Signature Example (RFC 9591 Compliant)
//!
//! This example demonstrates how two parties can jointly sign a message
//! using the FROST protocol without revealing their individual secret key shares.
//!
//! Run with: cargo run -p frost-signature

use frost_signature::frost::{
    aggregate_signature, keygen, sign_round1, sign_round2, verify_signature, FrostError,
    PublicShare,
};

fn main() -> Result<(), FrostError> {
    println!("=== FROST 2-of-2 Threshold Signature Demo (RFC 9591) ===\n");

    // Step 1: Key Generation (Shamir Secret Sharing)
    println!("Step 1: Key Generation (Shamir Secret Sharing)");
    println!("  Generating polynomial f(x) = s + a*x where s is the secret");
    let (share1, share2) = keygen();
    println!("  Party 1 receives share s_1 = f(1)");
    println!("  Party 2 receives share s_2 = f(2)");
    println!("  Group public key Y = s*G is shared by both parties");
    println!("  Secret s can be reconstructed via Lagrange interpolation\n");

    // The message to sign
    let message = "Transfer 100 tokens to Alice";
    println!("Message to sign: \"{}\"\n", message);

    // Step 2: Round 1 - Nonce Generation (TWO nonces per party)
    println!("Step 2: Round 1 - Nonce Commitments");
    println!("  Party 1 generates hiding nonce d_1 and binding nonce e_1");
    println!("  Party 1 computes commitments D_1 = d_1*G, E_1 = e_1*G");
    let (nonces1, commitment1) = sign_round1(&share1);
    println!("  Party 2 generates hiding nonce d_2 and binding nonce e_2");
    println!("  Party 2 computes commitments D_2 = d_2*G, E_2 = e_2*G");
    let (nonces2, commitment2) = sign_round1(&share2);
    println!("  Parties exchange commitment pairs (D_i, E_i)\n");

    // Simulate network exchange of commitments
    let all_commitments = vec![commitment1.clone(), commitment2.clone()];

    // Step 3: Round 2 - Partial Signature Generation
    println!("Step 3: Round 2 - Partial Signatures");
    println!("  Both parties compute binding factors rho_i from all commitments");
    println!("  This prevents manipulation of nonce contributions");
    println!("  Combined nonce: R = (D_1 + rho_1*E_1) + (D_2 + rho_2*E_2)");
    println!("  Challenge: c = H(R || Y || message)");
    println!("  Party 1 computes z_1 = d_1 + e_1*rho_1 + lambda_1*s_1*c");
    let partial1 = sign_round2(&share1, &nonces1, &all_commitments, message)?;
    println!("  Party 2 computes z_2 = d_2 + e_2*rho_2 + lambda_2*s_2*c");
    let partial2 = sign_round2(&share2, &nonces2, &all_commitments, message)?;
    println!("  (lambda_i are Lagrange coefficients: lambda_1=2, lambda_2=-1)\n");

    // Step 4: Aggregation
    println!("Step 4: Signature Aggregation");
    let public_shares = vec![
        PublicShare {
            identifier: share1.identifier,
            public_share: share1.public_share.clone(),
        },
        PublicShare {
            identifier: share2.identifier,
            public_share: share2.public_share.clone(),
        },
    ];

    let signature = aggregate_signature(
        &share1.group_public_key,
        &all_commitments,
        &[partial1, partial2],
        &public_shares,
        message,
    )?;
    println!("  Combined signature: (R, z) where z = z_1 + z_2\n");

    // Step 5: Verification
    println!("Step 5: Signature Verification");
    println!("  Verifier checks: z*G == R + c*Y");
    let is_valid = verify_signature(&share1.group_public_key, &signature, message)?;

    if is_valid {
        println!("  Signature is VALID!\n");
    } else {
        println!("  Signature is INVALID!\n");
    }

    // Demonstrate that tampering fails
    println!("=== Security Demonstration ===\n");
    println!("Attempting to verify with tampered message...");
    let tampered = "Transfer 1000 tokens to Alice";
    let tampered_valid = verify_signature(&share1.group_public_key, &signature, tampered)?;
    println!(
        "Tampered message verification: {}\n",
        if tampered_valid {
            "VALID (BAD!)"
        } else {
            "INVALID (GOOD!)"
        }
    );

    println!("=== Protocol Complete ===");
    println!("Key features of this RFC 9591 compliant implementation:");
    println!("  - Two nonces per party (hiding + binding) for security");
    println!("  - Binding factors prevent nonce manipulation attacks");
    println!("  - Lagrange coefficients enable threshold reconstruction");
    println!("  - Domain-separated hashing for cryptographic hygiene");
    println!("  - The signature is indistinguishable from standard Schnorr");

    Ok(())
}
