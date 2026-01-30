//! BLS Signature implementation using BLS12-381 curve.
//!
//! This module provides BLS (Boneh-Lynn-Shacham) signatures with support for:
//! - Basic signing and verification
//! - Signature aggregation (multiple signatures -> one signature)
//! - Public key aggregation (multiple public keys -> one public key)
//! - Batch verification
//!
//! # Security Warning
//!
//! This implementation is for **educational purposes only**. It is NOT suitable for
//! production use because:
//!
//! - The hash-to-curve implementation is simplified
//! - Operations may not be constant-time
//! - No protection against rogue key attacks in aggregation (would need proof-of-possession)
//!
//! For production use, consider well-audited libraries like `blst` or `bls-signatures`.

use lambdaworks_crypto::hash::sha3::Sha3Hasher;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::Degree2ExtensionField;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::{
    final_exponentiation, miller, BLS12381AtePairing,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
use lambdaworks_math::elliptic_curve::traits::{IsEllipticCurve, IsPairing};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::traits::ByteConversion;
use sha3::{Digest, Sha3_256};

/// Domain separation tag for BLS signatures (following IETF standard)
pub const BLS_SIG_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";

/// Type alias for G1 points (public keys)
pub type G1Point = ShortWeierstrassJacobianPoint<BLS12381Curve>;

/// Type alias for G2 points (signatures, hashed messages)
pub type G2Point = ShortWeierstrassJacobianPoint<BLS12381TwistCurve>;

/// Type alias for Fp2 elements
pub type Fp2Element = FieldElement<Degree2ExtensionField>;

/// BLS Secret Key
#[derive(Clone, Debug)]
pub struct SecretKey {
    /// The secret scalar
    sk: FrElement,
}

/// BLS Public Key (point in G1)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicKey {
    /// The public key point: pk = sk * G1
    pk: G1Point,
}

/// BLS Signature (point in G2)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// The signature point: sig = sk * H(m)
    sig: G2Point,
}

/// Errors that can occur during BLS operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlsError {
    /// Invalid secret key (zero)
    InvalidSecretKey,
    /// Invalid signature format
    InvalidSignature,
    /// Invalid public key format
    InvalidPublicKey,
    /// Signature verification failed
    VerificationFailed,
    /// Empty input for aggregation
    EmptyInput,
    /// Pairing computation error
    PairingError,
}

impl SecretKey {
    /// Create a new secret key from a scalar value.
    pub fn new(sk: FrElement) -> Result<Self, BlsError> {
        if sk == FrElement::zero() {
            return Err(BlsError::InvalidSecretKey);
        }
        Ok(Self { sk })
    }

    /// Create a secret key from bytes (32 bytes, big-endian).
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, BlsError> {
        let sk = FrElement::from_bytes_be(bytes).map_err(|_| BlsError::InvalidSecretKey)?;
        Self::new(sk)
    }

    /// Generate a deterministic secret key from a seed (for testing only).
    /// In production, use a cryptographically secure random number generator.
    pub fn from_seed(seed: &[u8]) -> Result<Self, BlsError> {
        let hash = Sha3_256::digest(seed);
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&hash);
        Self::from_bytes(&bytes)
    }

    /// Derive the corresponding public key.
    pub fn public_key(&self) -> PublicKey {
        let g1 = BLS12381Curve::generator();
        let pk = g1.operate_with_self(self.sk.representative());
        PublicKey { pk }
    }

    /// Sign a message.
    ///
    /// The signature is computed as: sig = sk * H(m)
    /// where H(m) is the hash of the message mapped to G2.
    pub fn sign(&self, message: &[u8]) -> Signature {
        let h = hash_to_g2(message);
        let sig = h.operate_with_self(self.sk.representative());
        Signature { sig }
    }

    /// Get the secret scalar (for advanced use cases).
    pub fn scalar(&self) -> &FrElement {
        &self.sk
    }
}

impl PublicKey {
    /// Create a public key from a G1 point.
    pub fn new(pk: G1Point) -> Result<Self, BlsError> {
        if pk.is_neutral_element() {
            return Err(BlsError::InvalidPublicKey);
        }
        Ok(Self { pk })
    }

    /// Verify a signature on a message.
    ///
    /// Verification checks: e(pk, H(m)) == e(G1, sig)
    /// This is equivalent to: e(pk, H(m)) * e(-G1, sig) == 1
    pub fn verify(&self, message: &[u8], signature: &Signature) -> Result<(), BlsError> {
        let h = hash_to_g2(message);
        let g1 = BLS12381Curve::generator();

        // Compute e(pk, H(m))
        let lhs = BLS12381AtePairing::compute_batch(&[(&self.pk.to_affine(), &h.to_affine())])
            .map_err(|_| BlsError::PairingError)?;

        // Compute e(G1, sig)
        let rhs =
            BLS12381AtePairing::compute_batch(&[(&g1.to_affine(), &signature.sig.to_affine())])
                .map_err(|_| BlsError::PairingError)?;

        if lhs == rhs {
            Ok(())
        } else {
            Err(BlsError::VerificationFailed)
        }
    }

    /// Get the underlying G1 point.
    pub fn point(&self) -> &G1Point {
        &self.pk
    }
}

impl Signature {
    /// Create a signature from a G2 point.
    pub fn new(sig: G2Point) -> Result<Self, BlsError> {
        if sig.is_neutral_element() {
            return Err(BlsError::InvalidSignature);
        }
        Ok(Self { sig })
    }

    /// Get the underlying G2 point.
    pub fn point(&self) -> &G2Point {
        &self.sig
    }

    /// Serialize the signature to bytes (96 bytes for compressed G2).
    pub fn to_bytes(&self) -> Vec<u8> {
        let affine = self.sig.to_affine();
        let [x, y, _] = affine.coordinates();
        let [x0, x1] = x.value();
        let [y0, _y1] = y.value();

        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(&x0.to_bytes_be());
        bytes.extend_from_slice(&x1.to_bytes_be());

        // Add sign bit for compression (simplified - just use y0's sign)
        let y0_bytes = y0.to_bytes_be();
        bytes[0] |= if y0_bytes[47] & 1 == 1 { 0x80 } else { 0x00 };

        bytes
    }
}

/// Hash a message to a point on G2.
///
/// This is a simplified hash-to-curve implementation. A production implementation
/// should follow the IETF hash-to-curve standard (RFC 9380).
pub fn hash_to_g2(message: &[u8]) -> G2Point {
    // Expand message using SHA3 with domain separation
    let expanded = Sha3Hasher::expand_message(message, BLS_SIG_DST, 128)
        .expect("expand_message should not fail with valid DST");

    // Convert first 8 bytes to a scalar for deterministic point generation
    let mut scalar_bytes = [0u8; 8];
    scalar_bytes.copy_from_slice(&expanded[0..8]);
    let scalar = u64::from_be_bytes(scalar_bytes);

    // Use generator scaled by hash-derived scalar
    // This is a simplified approach - production should use proper hash-to-curve
    let g2 = BLS12381TwistCurve::generator();
    g2.operate_with_self(scalar.saturating_add(1)) // Ensure non-zero
}

/// Convert hash bytes to an Fp2 element.
/// Note: This is kept for reference - a production implementation would use this
/// with a proper hash-to-curve implementation.
#[allow(dead_code)]
fn hash_bytes_to_fp2(bytes: &[u8]) -> Fp2Element {
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;

    // Convert bytes to hex strings and create field elements
    // This ensures proper modular reduction
    let hex0 = bytes_to_hex(&bytes[0..32]);
    let hex1 = bytes_to_hex(&bytes[32..64]);

    let fp0 = FieldElement::<BLS12381PrimeField>::from_hex_unchecked(&hex0);
    let fp1 = FieldElement::<BLS12381PrimeField>::from_hex_unchecked(&hex1);

    Fp2Element::new([fp0, fp1])
}

/// Convert bytes to hex string
#[allow(dead_code)]
fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Map an Fp2 element to a point on G2 (simplified).
///
/// This is a basic implementation that tries to find a valid y-coordinate.
/// A production implementation should use the optimized SWU map.
#[allow(dead_code)]
fn map_to_g2(u: &Fp2Element) -> G2Point {
    use lambdaworks_math::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;

    let b = BLS12381TwistCurve::b();

    // Try to find a point: iterate x = u, u+1, u+2, ... until we find a valid point
    let mut x = u.clone();
    for _ in 0..100 {
        // y^2 = x^3 + b
        let x_cubed = &x * &x * &x;
        let rhs = &x_cubed + &b;

        // Try to compute square root
        if let Some(y) = fp2_sqrt(&rhs) {
            if let Ok(point) = BLS12381TwistCurve::create_point_from_affine(x.clone(), y) {
                return point;
            }
        }

        // Increment x and try again
        x = &x + &Fp2Element::one();
    }

    // Fallback to generator (should not happen in practice)
    BLS12381TwistCurve::generator()
}

/// Compute square root in Fp2 (simplified Tonelli-Shanks).
#[allow(dead_code)]
fn fp2_sqrt(a: &Fp2Element) -> Option<Fp2Element> {
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
    use lambdaworks_math::unsigned_integer::element::U384;

    // For Fp2 = Fp[i]/(i^2 + 1), we use the formula:
    // sqrt(a + bi) requires solving a system
    // This is a simplified version that works for many cases

    let [a0, a1] = a.value();

    // If a1 = 0, just compute sqrt in Fp
    if *a1 == FieldElement::<BLS12381PrimeField>::zero() {
        // p = 3 mod 4 for BLS12-381, so sqrt(a) = a^((p+1)/4)
        // (p+1)/4 for BLS12-381
        let exp = U384::from_hex_unchecked(
            "0680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaab",
        );
        let sqrt_a0 = a0.pow(exp);

        // Verify
        if &sqrt_a0 * &sqrt_a0 == *a0 {
            return Some(Fp2Element::new([sqrt_a0, FieldElement::zero()]));
        }
        return None;
    }

    // For general case, use the complex sqrt formula
    // This is simplified and may not work for all inputs
    let norm = a0.square() + a1.square();
    let exp = U384::from_hex_unchecked(
        "0680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaab",
    );
    let sqrt_norm = norm.pow(exp);

    if &sqrt_norm * &sqrt_norm != norm {
        return None;
    }

    // alpha = (a0 + sqrt_norm) / 2
    let two_inv = FieldElement::<BLS12381PrimeField>::from(2u64)
        .inv()
        .unwrap();
    let alpha = (a0 + &sqrt_norm) * &two_inv;
    let sqrt_alpha = alpha.pow(exp);

    if &sqrt_alpha * &sqrt_alpha != alpha {
        // Try (a0 - sqrt_norm) / 2
        let alpha2 = (a0 - &sqrt_norm) * &two_inv;
        let sqrt_alpha2 = alpha2.pow(exp);
        if &sqrt_alpha2 * &sqrt_alpha2 == alpha2 {
            let two_sqrt_alpha2_inv = (&sqrt_alpha2 + &sqrt_alpha2).inv().ok()?;
            let beta = a1 * &two_sqrt_alpha2_inv;
            return Some(Fp2Element::new([sqrt_alpha2, beta]));
        }
        return None;
    }

    let two_sqrt_alpha_inv = (&sqrt_alpha + &sqrt_alpha).inv().ok()?;
    let beta = a1 * &two_sqrt_alpha_inv;

    Some(Fp2Element::new([sqrt_alpha, beta]))
}

/// Clear the G2 cofactor to get a point in the correct subgroup.
///
/// Uses the efficient endomorphism-based method for BLS12-381 G2 cofactor clearing.
/// The cofactor is h2 = (x^2 - x - 1) * (x - 1)^2 / 3 where x is the BLS12-381 parameter.
///
/// For simplicity, we use multiple scalar multiplications with smaller factors.
#[allow(dead_code)]
fn clear_cofactor_g2(p: &G2Point) -> G2Point {
    // The G2 cofactor h2 can be decomposed for efficient computation.
    // h2 = 305502333931268344200999753193121504214466019254188142667664032982267604182971884026507427359259977847832272839041692990889188039904403802465579155252111
    //
    // We use a simplified approach: multiply by x^2 - x - 1 then by (x-1)^2/3
    // where x = -0xd201000000010000 (the BLS12-381 parameter)
    //
    // For this educational example, we use the subgroup check instead.
    // A point is in the correct subgroup if h2 * P != identity.
    // Since we're starting from a valid hash, we just multiply by a safe scalar.

    // Use a simpler cofactor clearing: multiply by powers of x
    // x = 0xd201000000010000
    let x: u64 = 0xd201000000010000;

    // Compute [x]P
    let xp = p.operate_with_self(x);

    // Compute [x^2]P = [x]([x]P)
    let x2p = xp.operate_with_self(x);

    // Compute [x^2 - x - 1]P = [x^2]P - [x]P - P
    let neg_xp = xp.neg();
    let neg_p = p.neg();

    x2p.operate_with(&neg_xp).operate_with(&neg_p)
}

/// Aggregate multiple signatures into a single signature.
///
/// The aggregated signature is: sig_agg = sig_1 + sig_2 + ... + sig_n
///
/// # Security Note
/// This basic aggregation is vulnerable to rogue key attacks.
/// Production implementations should use proof-of-possession or
/// message augmentation schemes.
pub fn aggregate_signatures(signatures: &[Signature]) -> Result<Signature, BlsError> {
    if signatures.is_empty() {
        return Err(BlsError::EmptyInput);
    }

    let aggregated = signatures
        .iter()
        .map(|s| s.sig.clone())
        .reduce(|acc, sig| acc.operate_with(&sig))
        .ok_or(BlsError::EmptyInput)?;

    Ok(Signature { sig: aggregated })
}

/// Aggregate multiple public keys into a single public key.
///
/// The aggregated public key is: pk_agg = pk_1 + pk_2 + ... + pk_n
///
/// # Security Note
/// This basic aggregation is vulnerable to rogue key attacks.
/// Production implementations should use proof-of-possession.
pub fn aggregate_public_keys(public_keys: &[PublicKey]) -> Result<PublicKey, BlsError> {
    if public_keys.is_empty() {
        return Err(BlsError::EmptyInput);
    }

    let aggregated = public_keys
        .iter()
        .map(|pk| pk.pk.clone())
        .reduce(|acc, pk| acc.operate_with(&pk))
        .ok_or(BlsError::EmptyInput)?;

    Ok(PublicKey { pk: aggregated })
}

/// Verify an aggregated signature on a single message signed by multiple parties.
///
/// This verifies that the aggregated signature is valid for the message
/// under the aggregated public key.
pub fn verify_aggregated(
    message: &[u8],
    signature: &Signature,
    public_keys: &[PublicKey],
) -> Result<(), BlsError> {
    let aggregated_pk = aggregate_public_keys(public_keys)?;
    aggregated_pk.verify(message, signature)
}

/// Batch verify multiple signatures on different messages.
///
/// This is more efficient than verifying each signature individually
/// when you have multiple (message, signature, public key) tuples.
///
/// Returns Ok(()) if ALL signatures are valid, Err otherwise.
pub fn batch_verify(
    messages: &[&[u8]],
    signatures: &[Signature],
    public_keys: &[PublicKey],
) -> Result<(), BlsError> {
    if messages.len() != signatures.len() || messages.len() != public_keys.len() {
        return Err(BlsError::EmptyInput);
    }

    if messages.is_empty() {
        return Err(BlsError::EmptyInput);
    }

    // For batch verification, we check:
    // e(pk_1, H(m_1)) * e(pk_2, H(m_2)) * ... == e(G1, sig_1 + sig_2 + ...)
    //
    // This is equivalent to checking:
    // product of e(pk_i, H(m_i)) == e(G1, sum of sig_i)

    let g1 = BLS12381Curve::generator();

    // Compute sum of signatures
    let sig_sum = signatures
        .iter()
        .map(|s| s.sig.clone())
        .reduce(|acc, sig| acc.operate_with(&sig))
        .ok_or(BlsError::EmptyInput)?;

    // Compute product of pairings e(pk_i, H(m_i))
    let mut lhs = FieldElement::one();
    for (pk, msg) in public_keys.iter().zip(messages.iter()) {
        let h = hash_to_g2(msg);
        let f = miller(&h.to_affine(), &pk.pk.to_affine());
        lhs *= f;
    }
    let lhs = final_exponentiation(&lhs);

    // Compute e(G1, sig_sum)
    let rhs = BLS12381AtePairing::compute_batch(&[(&g1.to_affine(), &sig_sum.to_affine())])
        .map_err(|_| BlsError::PairingError)?;

    if lhs == rhs {
        Ok(())
    } else {
        Err(BlsError::VerificationFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let sk = SecretKey::from_seed(b"test seed 1").unwrap();
        let pk = sk.public_key();
        assert!(!pk.pk.is_neutral_element());
    }

    #[test]
    fn test_sign_and_verify() {
        let sk = SecretKey::from_seed(b"test seed").unwrap();
        let pk = sk.public_key();
        let message = b"Hello, BLS!";

        let signature = sk.sign(message);
        assert!(pk.verify(message, &signature).is_ok());
    }

    #[test]
    fn test_wrong_message_fails() {
        let sk = SecretKey::from_seed(b"test seed").unwrap();
        let pk = sk.public_key();

        let signature = sk.sign(b"message 1");
        assert!(pk.verify(b"message 2", &signature).is_err());
    }

    #[test]
    fn test_wrong_key_fails() {
        let sk1 = SecretKey::from_seed(b"test seed 1").unwrap();
        let sk2 = SecretKey::from_seed(b"test seed 2").unwrap();
        let pk2 = sk2.public_key();
        let message = b"Hello, BLS!";

        let signature = sk1.sign(message);
        assert!(pk2.verify(message, &signature).is_err());
    }

    #[test]
    fn test_signature_aggregation() {
        let sk1 = SecretKey::from_seed(b"signer 1").unwrap();
        let sk2 = SecretKey::from_seed(b"signer 2").unwrap();
        let pk1 = sk1.public_key();
        let pk2 = sk2.public_key();

        let message = b"Shared message";

        // Both sign the same message
        let sig1 = sk1.sign(message);
        let sig2 = sk2.sign(message);

        // Aggregate signatures
        let agg_sig = aggregate_signatures(&[sig1, sig2]).unwrap();

        // Verify aggregated signature with aggregated public key
        let result = verify_aggregated(message, &agg_sig, &[pk1, pk2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_verification() {
        let sk1 = SecretKey::from_seed(b"signer 1").unwrap();
        let sk2 = SecretKey::from_seed(b"signer 2").unwrap();
        let pk1 = sk1.public_key();
        let pk2 = sk2.public_key();

        let msg1 = b"Message 1";
        let msg2 = b"Message 2";

        let sig1 = sk1.sign(msg1);
        let sig2 = sk2.sign(msg2);

        // Batch verify
        let result = batch_verify(
            &[msg1.as_slice(), msg2.as_slice()],
            &[sig1, sig2],
            &[pk1, pk2],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_secret_key() {
        let result = SecretKey::new(FrElement::zero());
        assert_eq!(result.unwrap_err(), BlsError::InvalidSecretKey);
    }

    #[test]
    fn test_aggregate_empty_fails() {
        let result = aggregate_signatures(&[]);
        assert_eq!(result.unwrap_err(), BlsError::EmptyInput);
    }

    #[test]
    fn test_three_party_aggregation() {
        let sk1 = SecretKey::from_seed(b"party 1").unwrap();
        let sk2 = SecretKey::from_seed(b"party 2").unwrap();
        let sk3 = SecretKey::from_seed(b"party 3").unwrap();

        let pk1 = sk1.public_key();
        let pk2 = sk2.public_key();
        let pk3 = sk3.public_key();

        let message = b"Multi-party consensus";

        let sig1 = sk1.sign(message);
        let sig2 = sk2.sign(message);
        let sig3 = sk3.sign(message);

        let agg_sig = aggregate_signatures(&[sig1, sig2, sig3]).unwrap();

        let result = verify_aggregated(message, &agg_sig, &[pk1, pk2, pk3]);
        assert!(result.is_ok());
    }
}
