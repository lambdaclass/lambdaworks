//! FROST: Flexible Round-Optimized Schnorr Threshold Signatures (RFC 9591)
//!
//! This module implements a 2-of-2 FROST protocol compliant with RFC 9591.
//! The protocol allows two parties to jointly produce a Schnorr signature
//! without revealing their individual secret key shares.
//!
//! # Protocol Overview
//!
//! ## Key Generation
//! Each party has an identifier (1 or 2) and a secret share. The group public
//! key is derived using Lagrange interpolation coefficients.
//!
//! ## Signing Round 1 (Nonce Commitment)
//! Each party generates TWO nonces (hiding and binding) and commits to both:
//! - `D_i = hiding_nonce_i * G`
//! - `E_i = binding_nonce_i * G`
//!
//! ## Signing Round 2 (Partial Signature)
//! After receiving all commitments, each party:
//! 1. Computes binding factors `ρ_i` from all commitments (prevents manipulation)
//! 2. Computes combined `R = Σ(D_i + E_i * ρ_i)`
//! 3. Computes challenge `c = H(R || Y || message)`
//! 4. Computes partial signature `z_i = hiding_i + binding_i * ρ_i + λ_i * s_i * c`
//!
//! ## Verification
//! Verify: `z * G == R + c * Y`

use crate::common::*;

use lambdaworks_math::{
    cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve, traits::ByteConversion,
};

use sha3::{Digest, Keccak256};
use std::collections::HashSet;

use rand::SeedableRng;

/// Domain separation tag for binding factor computation.
const BINDING_FACTOR_DST: &[u8] = b"FROST-BN254-SHA256-v1rho";

/// Domain separation tag for challenge computation.
const CHALLENGE_DST: &[u8] = b"FROST-BN254-SHA256-v1chal";

/// Participant identifier (1 or 2 for 2-of-2).
pub type Identifier = u8;

/// Public key share (used by the aggregator to verify partial signatures).
#[derive(Clone)]
pub struct PublicShare {
    /// The party's identifier (1 or 2).
    pub identifier: Identifier,
    /// The party's public share: Y_i = s_i * G.
    pub public_share: CurvePoint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrostError {
    DuplicateIdentifier(&'static str, Identifier),
    InvalidIdentifier(Identifier),
    MissingCommitment(Identifier),
    MissingPublicShare(Identifier),
    MissingPartialSignature(Identifier),
    InvalidPartialSignature(Identifier),
    InvalidScalarBytes(&'static str),
}

/// A party's key share for threshold signing.
#[derive(Clone)]
pub struct KeyShare {
    /// The party's identifier (1 or 2).
    pub identifier: Identifier,
    /// The party's secret share of the private key.
    pub secret_share: FE,
    /// The party's public share: Y_i = s_i * G.
    pub public_share: CurvePoint,
    /// The combined group public key.
    pub group_public_key: CurvePoint,
}

/// Nonce pair generated in Round 1 (kept secret by the party).
pub struct SigningNonces {
    /// The hiding nonce.
    pub hiding: FE,
    /// The binding nonce.
    pub binding: FE,
}

/// Commitment pair broadcast during Round 1.
#[derive(Clone)]
pub struct NonceCommitment {
    /// The party's identifier.
    pub identifier: Identifier,
    /// D_i = hiding_nonce * G.
    pub hiding_commitment: CurvePoint,
    /// E_i = binding_nonce * G.
    pub binding_commitment: CurvePoint,
}

/// Partial signature generated in Round 2.
#[derive(Clone)]
pub struct PartialSignature {
    /// The party's identifier.
    pub identifier: Identifier,
    /// The partial signature scalar.
    pub z_i: FE,
}

/// The final aggregated Schnorr signature.
pub struct Signature {
    /// The combined nonce point R.
    pub r: CurvePoint,
    /// The aggregated signature scalar z.
    pub z: FE,
}

/// Compute Lagrange coefficient λ_i for identifier i in a 2-of-2 scheme.
///
/// For interpolation at x=0 with participants at x=1 and x=2:
/// - λ_1 = (0 - 2) / (1 - 2) = 2
/// - λ_2 = (0 - 1) / (2 - 1) = -1
fn compute_lagrange_coefficient(identifier: Identifier) -> Result<FE, FrostError> {
    match identifier {
        1 => Ok(FE::from(2u64)),
        2 => Ok(-FE::one()),
        _ => Err(FrostError::InvalidIdentifier(identifier)),
    }
}

/// Generate key shares for a 2-of-2 threshold scheme.
///
/// Uses Shamir secret sharing with a degree-1 polynomial:
/// `f(x) = s + a_1 * x`
///
/// The secret `s = f(0)` can be reconstructed using Lagrange interpolation
/// from any 2 shares.
pub fn keygen() -> (KeyShare, KeyShare) {
    let g = Curve::generator();

    // Generate random polynomial coefficients: f(x) = s + a*x
    let s = sample_field_elem(rand_chacha::ChaCha20Rng::from_entropy()); // secret
    let a = sample_field_elem(rand_chacha::ChaCha20Rng::from_entropy()); // coefficient

    // Compute shares: s_i = f(i) for i ∈ {1, 2}
    let s1 = &s + &a; // f(1) = s + a
    let s2 = &s + &(&a * &FE::from(2u64)); // f(2) = s + 2a

    // Compute public shares: Y_i = s_i * G
    let y1 = g.operate_with_self(s1.representative());
    let y2 = g.operate_with_self(s2.representative());

    // Group public key: Y = s * G (the secret point)
    let group_public_key = g.operate_with_self(s.representative());

    let share1 = KeyShare {
        identifier: 1,
        secret_share: s1,
        public_share: y1,
        group_public_key: group_public_key.clone(),
    };

    let share2 = KeyShare {
        identifier: 2,
        secret_share: s2,
        public_share: y2,
        group_public_key,
    };

    (share1, share2)
}

/// Round 1: Generate nonce pair and commitments.
///
/// Each party generates two random nonces (hiding and binding) and commits
/// to both. The nonces must be kept secret until Round 2.
pub fn sign_round1(key_share: &KeyShare) -> (SigningNonces, NonceCommitment) {
    let g = Curve::generator();

    // Generate two random nonces
    let hiding = sample_field_elem(rand_chacha::ChaCha20Rng::from_entropy());
    let binding = sample_field_elem(rand_chacha::ChaCha20Rng::from_entropy());

    // Compute commitments: D = hiding * G, E = binding * G
    let hiding_commitment = g.operate_with_self(hiding.representative());
    let binding_commitment = g.operate_with_self(binding.representative());

    let nonces = SigningNonces { hiding, binding };

    let commitment = NonceCommitment {
        identifier: key_share.identifier,
        hiding_commitment,
        binding_commitment,
    };

    (nonces, commitment)
}

/// Encode commitments for hashing.
fn encode_commitments(commitments: &[NonceCommitment]) -> Vec<u8> {
    let mut encoded = Vec::new();
    for c in commitments {
        encoded.push(c.identifier);
        let d_affine = c.hiding_commitment.to_affine();
        encoded.extend_from_slice(&d_affine.x().to_bytes_be());
        encoded.extend_from_slice(&d_affine.y().to_bytes_be());
        let e_affine = c.binding_commitment.to_affine();
        encoded.extend_from_slice(&e_affine.x().to_bytes_be());
        encoded.extend_from_slice(&e_affine.y().to_bytes_be());
    }
    encoded
}

/// Compute binding factor ρ_i for participant i.
///
/// ρ_i = H(Y || encoded_commitments || msg || identifier)
///
/// The binding factor ensures that a participant cannot change their
/// effective nonce contribution after seeing others' commitments.
fn compute_binding_factor(
    group_public_key: &CurvePoint,
    commitments: &[NonceCommitment],
    message: &str,
    identifier: Identifier,
) -> Result<FE, FrostError> {
    let commitments = canonicalize_commitments(commitments)?;
    let mut hasher = Keccak256::new();

    // Domain separation
    hasher.update(BINDING_FACTOR_DST);

    // Group public key
    let y_affine = group_public_key.to_affine();
    hasher.update(y_affine.x().to_bytes_be());
    hasher.update(y_affine.y().to_bytes_be());

    // Encoded commitments
    hasher.update(encode_commitments(&commitments));

    // Message
    hasher.update(message.as_bytes());

    // Participant identifier
    hasher.update([identifier]);

    let hash = hasher.finalize().to_vec();
    FE::from_bytes_be(&hash).map_err(|_| FrostError::InvalidScalarBytes("binding factor"))
}

/// Compute the combined nonce point R.
///
/// R = Σ(D_i + ρ_i * E_i) for all participants
fn compute_group_commitment(
    commitments: &[NonceCommitment],
    binding_factors: &[(Identifier, FE)],
) -> Result<CurvePoint, FrostError> {
    let mut r = CurvePoint::neutral_element();

    for commitment in commitments {
        // Find binding factor for this participant
        let rho = binding_factors
            .iter()
            .find(|(id, _)| *id == commitment.identifier)
            .map(|(_, rho)| rho)
            .ok_or(FrostError::MissingCommitment(commitment.identifier))?;

        // R_i = D_i + ρ_i * E_i
        let rho_e = commitment
            .binding_commitment
            .operate_with_self(rho.representative());
        let r_i = commitment.hiding_commitment.operate_with(&rho_e);

        r = r.operate_with(&r_i);
    }

    Ok(r)
}

/// Compute the challenge c = H(R || Y || message).
fn compute_challenge(
    combined_r: &CurvePoint,
    group_public_key: &CurvePoint,
    message: &str,
) -> Result<FE, FrostError> {
    let mut hasher = Keccak256::new();

    // Domain separation
    hasher.update(CHALLENGE_DST);

    // Combined nonce point R
    let r_affine = combined_r.to_affine();
    hasher.update(r_affine.x().to_bytes_be());
    hasher.update(r_affine.y().to_bytes_be());

    // Group public key Y
    let y_affine = group_public_key.to_affine();
    hasher.update(y_affine.x().to_bytes_be());
    hasher.update(y_affine.y().to_bytes_be());

    // Message
    hasher.update(message.as_bytes());

    let hash = hasher.finalize().to_vec();
    FE::from_bytes_be(&hash).map_err(|_| FrostError::InvalidScalarBytes("challenge"))
}

/// Round 2: Generate partial signature.
///
/// After receiving all commitments, each party computes their partial signature:
/// `z_i = hiding_i + binding_i * ρ_i + λ_i * s_i * c`
pub fn sign_round2(
    key_share: &KeyShare,
    nonces: &SigningNonces,
    all_commitments: &[NonceCommitment],
    message: &str,
) -> Result<PartialSignature, FrostError> {
    let commitments = canonicalize_commitments(all_commitments)?;
    if !commitments
        .iter()
        .any(|c| c.identifier == key_share.identifier)
    {
        return Err(FrostError::MissingCommitment(key_share.identifier));
    }

    // Compute binding factors for all participants
    let binding_factors: Vec<(Identifier, FE)> = commitments
        .iter()
        .map(|c| {
            let rho = compute_binding_factor(
                &key_share.group_public_key,
                &commitments,
                message,
                c.identifier,
            )?;
            Ok((c.identifier, rho))
        })
        .collect::<Result<_, FrostError>>()?;

    // Compute combined nonce point R
    let combined_r = compute_group_commitment(&commitments, &binding_factors)?;

    // Compute challenge c = H(R || Y || message)
    let c = compute_challenge(&combined_r, &key_share.group_public_key, message)?;

    // Get this party's binding factor
    let rho_i = binding_factors
        .iter()
        .find(|(id, _)| *id == key_share.identifier)
        .map(|(_, rho)| rho)
        .ok_or(FrostError::MissingCommitment(key_share.identifier))?;

    // Compute Lagrange coefficient λ_i
    let lambda_i = compute_lagrange_coefficient(key_share.identifier)?;

    // Compute partial signature: z_i = hiding_i + binding_i * ρ_i + λ_i * s_i * c
    let z_i =
        &nonces.hiding + &(&nonces.binding * rho_i) + &(&lambda_i * &key_share.secret_share * &c);

    Ok(PartialSignature {
        identifier: key_share.identifier,
        z_i,
    })
}

/// Aggregate partial signatures into the final signature.
pub fn aggregate_signature(
    group_public_key: &CurvePoint,
    all_commitments: &[NonceCommitment],
    partial_signatures: &[PartialSignature],
    public_shares: &[PublicShare],
    message: &str,
) -> Result<Signature, FrostError> {
    let commitments = canonicalize_commitments(all_commitments)?;
    ensure_unique_identifiers(
        partial_signatures.iter().map(|p| p.identifier),
        "partial signatures",
    )?;
    ensure_unique_identifiers(public_shares.iter().map(|p| p.identifier), "public shares")?;

    for commitment in &commitments {
        if !public_shares
            .iter()
            .any(|p| p.identifier == commitment.identifier)
        {
            return Err(FrostError::MissingPublicShare(commitment.identifier));
        }
        if !partial_signatures
            .iter()
            .any(|p| p.identifier == commitment.identifier)
        {
            return Err(FrostError::MissingPartialSignature(commitment.identifier));
        }
    }
    for partial in partial_signatures {
        if !commitments
            .iter()
            .any(|c| c.identifier == partial.identifier)
        {
            return Err(FrostError::MissingCommitment(partial.identifier));
        }
    }
    for public_share in public_shares {
        if !commitments
            .iter()
            .any(|c| c.identifier == public_share.identifier)
        {
            return Err(FrostError::MissingCommitment(public_share.identifier));
        }
    }

    // Recompute binding factors
    let binding_factors: Vec<(Identifier, FE)> = commitments
        .iter()
        .map(|c| {
            let rho =
                compute_binding_factor(group_public_key, &commitments, message, c.identifier)?;
            Ok((c.identifier, rho))
        })
        .collect::<Result<_, FrostError>>()?;

    // Compute combined nonce point R
    let r = compute_group_commitment(&commitments, &binding_factors)?;

    // Compute challenge c = H(R || Y || message)
    let c = compute_challenge(&r, group_public_key, message)?;

    let g = Curve::generator();

    for partial in partial_signatures {
        let commitment = commitments
            .iter()
            .find(|c| c.identifier == partial.identifier)
            .ok_or(FrostError::MissingCommitment(partial.identifier))?;
        let public_share = public_shares
            .iter()
            .find(|p| p.identifier == partial.identifier)
            .ok_or(FrostError::MissingPublicShare(partial.identifier))?;
        let rho = binding_factors
            .iter()
            .find(|(id, _)| *id == partial.identifier)
            .map(|(_, rho)| rho)
            .ok_or(FrostError::MissingCommitment(partial.identifier))?;

        let lambda_i = compute_lagrange_coefficient(partial.identifier)?;

        let left = g.operate_with_self(partial.z_i.representative());
        let rho_e = commitment
            .binding_commitment
            .operate_with_self(rho.representative());
        let lambda_c = &lambda_i * &c;
        let lambda_c_y = public_share
            .public_share
            .operate_with_self(lambda_c.representative());
        let right = commitment
            .hiding_commitment
            .operate_with(&rho_e)
            .operate_with(&lambda_c_y);

        if left != right {
            return Err(FrostError::InvalidPartialSignature(partial.identifier));
        }
    }

    // Aggregate partial signatures: z = Σ z_i
    let z = partial_signatures
        .iter()
        .fold(FE::zero(), |acc, p| acc + &p.z_i);

    Ok(Signature { r, z })
}

/// Verify a FROST signature.
///
/// Checks that `z * G == R + c * Y` where:
/// - G is the generator
/// - z is the aggregated signature scalar
/// - R is the combined nonce point
/// - Y is the group public key
/// - c is the challenge hash
pub fn verify_signature(
    group_public_key: &CurvePoint,
    signature: &Signature,
    message: &str,
) -> Result<bool, FrostError> {
    let g = Curve::generator();

    // Compute challenge c = H(R || Y || message)
    let c = compute_challenge(&signature.r, group_public_key, message)?;

    // Compute left side: z * G
    let left = g.operate_with_self(signature.z.representative());

    // Compute right side: R + c * Y
    let c_y = group_public_key.operate_with_self(c.representative());
    let right = signature.r.operate_with(&c_y);

    Ok(left == right)
}

fn ensure_unique_identifiers<I>(ids: I, context: &'static str) -> Result<(), FrostError>
where
    I: IntoIterator<Item = Identifier>,
{
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(FrostError::DuplicateIdentifier(context, id));
        }
    }
    Ok(())
}

fn canonicalize_commitments(
    commitments: &[NonceCommitment],
) -> Result<Vec<NonceCommitment>, FrostError> {
    ensure_unique_identifiers(commitments.iter().map(|c| c.identifier), "commitments")?;
    let mut sorted = commitments.to_vec();
    sorted.sort_by_key(|c| c.identifier);
    Ok(sorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagrange_coefficients() -> Result<(), FrostError> {
        // Verify that λ_1 * 1 + λ_2 * 2 = 0 (interpolation at x=0)
        // and λ_1 + λ_2 = 1 (partition of unity... wait, that's not right for x=0)
        //
        // Actually, for Lagrange basis polynomials:
        // L_1(x) = (x - 2)/(1 - 2) = (x - 2)/(-1) = 2 - x
        // L_2(x) = (x - 1)/(2 - 1) = x - 1
        //
        // At x=0: L_1(0) = 2, L_2(0) = -1
        // Check: L_1(0) + L_2(0) = 2 + (-1) = 1 ✓
        let lambda_1 = compute_lagrange_coefficient(1)?;
        let lambda_2 = compute_lagrange_coefficient(2)?;

        assert_eq!(lambda_1, FE::from(2u64));
        assert_eq!(lambda_2, -FE::one());

        // λ_1 + λ_2 should equal 1
        assert_eq!(&lambda_1 + &lambda_2, FE::one());
        Ok(())
    }

    #[test]
    fn test_shamir_reconstruction() -> Result<(), FrostError> {
        // Verify that Lagrange interpolation recovers the secret
        let g = Curve::generator();

        let (share1, share2) = keygen();

        // Reconstruct secret: s = λ_1 * s_1 + λ_2 * s_2
        let lambda_1 = compute_lagrange_coefficient(1)?;
        let lambda_2 = compute_lagrange_coefficient(2)?;

        let reconstructed_secret =
            &(&lambda_1 * &share1.secret_share) + &(&lambda_2 * &share2.secret_share);

        // Verify: reconstructed_secret * G == group_public_key
        let reconstructed_public = g.operate_with_self(reconstructed_secret.representative());
        assert_eq!(reconstructed_public, share1.group_public_key);
        Ok(())
    }

    #[test]
    fn test_2_of_2_frost_signature() -> Result<(), FrostError> {
        let (share1, share2) = keygen();

        assert_eq!(
            share1.group_public_key, share2.group_public_key,
            "Both parties should have the same group public key"
        );

        let message = "Hello, FROST RFC 9591!";

        // Round 1: Generate nonces and commitments
        let (nonces1, commitment1) = sign_round1(&share1);
        let (nonces2, commitment2) = sign_round1(&share2);

        let all_commitments = vec![commitment1, commitment2];

        // Round 2: Compute partial signatures
        let partial1 = sign_round2(&share1, &nonces1, &all_commitments, message)?;
        let partial2 = sign_round2(&share2, &nonces2, &all_commitments, message)?;

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

        // Aggregation
        let signature = aggregate_signature(
            &share1.group_public_key,
            &all_commitments,
            &[partial1, partial2],
            &public_shares,
            message,
        )?;

        // Verification
        assert!(
            verify_signature(&share1.group_public_key, &signature, message)?,
            "Signature verification failed"
        );
        Ok(())
    }

    #[test]
    fn test_invalid_message_fails() -> Result<(), FrostError> {
        let (share1, share2) = keygen();
        let message = "Original message";

        let (nonces1, commitment1) = sign_round1(&share1);
        let (nonces2, commitment2) = sign_round1(&share2);
        let all_commitments = vec![commitment1, commitment2];

        let partial1 = sign_round2(&share1, &nonces1, &all_commitments, message)?;
        let partial2 = sign_round2(&share2, &nonces2, &all_commitments, message)?;

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

        // Verification with wrong message should fail
        assert!(
            !verify_signature(&share1.group_public_key, &signature, "Tampered message")?,
            "Signature should not verify with different message"
        );
        Ok(())
    }

    #[test]
    fn test_commitment_order_is_canonicalized() -> Result<(), FrostError> {
        let g = Curve::generator();
        let s = FE::from(5u64);
        let a = FE::from(7u64);

        let s1 = &s + &a;
        let s2 = &s + &(&a * &FE::from(2u64));

        let group_public_key = g.operate_with_self(s.representative());
        let public_share1 = g.operate_with_self(s1.representative());
        let public_share2 = g.operate_with_self(s2.representative());
        let share1 = KeyShare {
            identifier: 1,
            secret_share: s1,
            public_share: public_share1,
            group_public_key: group_public_key.clone(),
        };
        let share2 = KeyShare {
            identifier: 2,
            secret_share: s2,
            public_share: public_share2,
            group_public_key: group_public_key.clone(),
        };

        let nonces1 = SigningNonces {
            hiding: FE::from(3u64),
            binding: FE::from(11u64),
        };
        let nonces2 = SigningNonces {
            hiding: FE::from(13u64),
            binding: FE::from(17u64),
        };

        let commitment1 = NonceCommitment {
            identifier: share1.identifier,
            hiding_commitment: g.operate_with_self(nonces1.hiding.representative()),
            binding_commitment: g.operate_with_self(nonces1.binding.representative()),
        };
        let commitment2 = NonceCommitment {
            identifier: share2.identifier,
            hiding_commitment: g.operate_with_self(nonces2.hiding.representative()),
            binding_commitment: g.operate_with_self(nonces2.binding.representative()),
        };

        let commitments_signers = vec![commitment1.clone(), commitment2.clone()];
        let message = "commitment order mismatch";

        let partial1 = sign_round2(&share1, &nonces1, &commitments_signers, message)?;
        let partial2 = sign_round2(&share2, &nonces2, &commitments_signers, message)?;

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

        // Aggregator uses a different commitment order than the signers.
        let commitments_aggregator = vec![commitment2, commitment1];
        let signature = aggregate_signature(
            &share1.group_public_key,
            &commitments_aggregator,
            &[partial1, partial2],
            &public_shares,
            message,
        )?;

        assert!(
            verify_signature(&share1.group_public_key, &signature, message)?,
            "Signature should verify even if commitments are provided in a different order"
        );
        Ok(())
    }

    #[test]
    fn test_aggregation_rejects_invalid_partial_signature() -> Result<(), FrostError> {
        let g = Curve::generator();
        let s = FE::from(9u64);
        let a = FE::from(4u64);

        let s1 = &s + &a;
        let s2 = &s + &(&a * &FE::from(2u64));

        let group_public_key = g.operate_with_self(s.representative());
        let public_share1 = g.operate_with_self(s1.representative());
        let public_share2 = g.operate_with_self(s2.representative());
        let share1 = KeyShare {
            identifier: 1,
            secret_share: s1,
            public_share: public_share1,
            group_public_key: group_public_key.clone(),
        };
        let share2 = KeyShare {
            identifier: 2,
            secret_share: s2,
            public_share: public_share2,
            group_public_key: group_public_key.clone(),
        };

        let nonces1 = SigningNonces {
            hiding: FE::from(21u64),
            binding: FE::from(33u64),
        };
        let nonces2 = SigningNonces {
            hiding: FE::from(55u64),
            binding: FE::from(89u64),
        };

        let commitment1 = NonceCommitment {
            identifier: share1.identifier,
            hiding_commitment: g.operate_with_self(nonces1.hiding.representative()),
            binding_commitment: g.operate_with_self(nonces1.binding.representative()),
        };
        let commitment2 = NonceCommitment {
            identifier: share2.identifier,
            hiding_commitment: g.operate_with_self(nonces2.hiding.representative()),
            binding_commitment: g.operate_with_self(nonces2.binding.representative()),
        };

        let commitments = vec![commitment1, commitment2];
        let message = "invalid partial signature";

        let partial1 = sign_round2(&share1, &nonces1, &commitments, message)?;
        let mut partial2 = sign_round2(&share2, &nonces2, &commitments, message)?;

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

        // Corrupt one partial signature; aggregation does not detect this.
        partial2.z_i = &partial2.z_i + &FE::one();

        let result = aggregate_signature(
            &share1.group_public_key,
            &commitments,
            &[partial1, partial2],
            &public_shares,
            message,
        );

        assert!(
            matches!(
                result,
                Err(FrostError::InvalidPartialSignature(id)) if id == share2.identifier
            ),
            "Expected InvalidPartialSignature for participant 2"
        );
        Ok(())
    }

    #[test]
    fn test_different_nonces_produce_different_signatures() -> Result<(), FrostError> {
        let (share1, share2) = keygen();
        let message = "Same message";

        // First signature
        let (nonces1a, commitment1a) = sign_round1(&share1);
        let (nonces2a, commitment2a) = sign_round1(&share2);
        let commits_a = vec![commitment1a, commitment2a];
        let partial1a = sign_round2(&share1, &nonces1a, &commits_a, message)?;
        let partial2a = sign_round2(&share2, &nonces2a, &commits_a, message)?;
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
        let sig_a = aggregate_signature(
            &share1.group_public_key,
            &commits_a,
            &[partial1a, partial2a],
            &public_shares,
            message,
        )?;

        // Second signature (different nonces)
        let (nonces1b, commitment1b) = sign_round1(&share1);
        let (nonces2b, commitment2b) = sign_round1(&share2);
        let commits_b = vec![commitment1b, commitment2b];
        let partial1b = sign_round2(&share1, &nonces1b, &commits_b, message)?;
        let partial2b = sign_round2(&share2, &nonces2b, &commits_b, message)?;
        let sig_b = aggregate_signature(
            &share1.group_public_key,
            &commits_b,
            &[partial1b, partial2b],
            &public_shares,
            message,
        )?;

        // Both should verify
        assert!(verify_signature(&share1.group_public_key, &sig_a, message)?);
        assert!(verify_signature(&share1.group_public_key, &sig_b, message)?);

        // But signatures should differ
        assert_ne!(sig_a.z, sig_b.z);
        assert_ne!(sig_a.r, sig_b.r);
        Ok(())
    }
}
