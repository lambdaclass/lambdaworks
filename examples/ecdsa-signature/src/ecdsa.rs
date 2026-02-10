//! ECDSA implementation for secp256k1.
//!
//! This module provides ECDSA signing and verification using the secp256k1 curve.
//!
//! # Security Warning
//!
//! This implementation is for **educational purposes only**. It is NOT suitable for
//! production use because:
//!
//! - Operations are NOT constant-time (vulnerable to timing attacks)
//! - Nonce generation is the caller's responsibility (must use RFC 6979 or CSPRNG)
//! - No protection against fault attacks
//! - Private keys and nonces are not zeroized after use (use `zeroize` crate in production)
//!
//! For production use, consider well-audited libraries like `k256` or `secp256k1`.

use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::secp256k1::curve::Secp256k1Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::secp256k1_scalarfield::Secp256k1ScalarField;
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::unsigned_integer::element::U256;

/// Type alias for scalar field elements
pub type ScalarFE = FieldElement<Secp256k1ScalarField>;

/// Type alias for curve points
pub type CurvePoint = ShortWeierstrassProjectivePoint<Secp256k1Curve>;

/// ECDSA signature containing r and s components
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// The x-coordinate of the random point R = k*G (mod n)
    pub r: ScalarFE,
    /// The signature proof s = k^(-1) * (z + r*d) (mod n)
    pub s: ScalarFE,
}

/// Errors that can occur during ECDSA operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EcdsaError {
    /// The signature r component is zero or out of range
    InvalidRValue,
    /// The signature s component is zero or out of range
    InvalidSValue,
    /// The nonce k is invalid (zero or >= order)
    InvalidNonce,
    /// Failed to compute the inverse
    InverseError,
    /// The message hash is invalid
    InvalidMessageHash,
    /// Signature verification failed
    VerificationFailed,
    /// Public key is not a valid curve point
    InvalidPublicKey,
}

/// Half of the curve order n, used for low-S normalization.
/// n/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
fn half_order() -> U256 {
    U256::from_hex_unchecked("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0")
}

/// Check if a point is on the secp256k1 curve.
/// Verifies that y² = x³ + 7 (mod p).
fn is_point_on_curve(point: &CurvePoint) -> bool {
    use lambdaworks_math::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;

    // Point at infinity is technically on the curve
    if *point == CurvePoint::neutral_element() {
        return true;
    }

    let affine = point.to_affine();
    let x = affine.x();
    let y = affine.y();

    // Check curve equation: y² = x³ + b (where b = 7 for secp256k1)
    let y_sq = y * y;
    let x_cubed = x * x * x;
    let rhs = x_cubed + Secp256k1Curve::b();

    y_sq == rhs
}

impl Signature {
    /// Create a new signature from r and s values.
    ///
    /// Returns an error if r or s is zero.
    pub fn new(r: ScalarFE, s: ScalarFE) -> Result<Self, EcdsaError> {
        if r == ScalarFE::zero() {
            return Err(EcdsaError::InvalidRValue);
        }
        if s == ScalarFE::zero() {
            return Err(EcdsaError::InvalidSValue);
        }
        Ok(Self { r, s })
    }

    /// Serialize the signature to bytes (64 bytes: 32 for r, 32 for s).
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        let r_bytes = self.r.to_bytes_be();
        let s_bytes = self.s.to_bytes_be();

        bytes[..32].copy_from_slice(&r_bytes);
        bytes[32..].copy_from_slice(&s_bytes);
        bytes
    }

    /// Deserialize a signature from bytes.
    pub fn from_bytes(bytes: &[u8; 64]) -> Result<Self, EcdsaError> {
        let r = ScalarFE::from_bytes_be(&bytes[..32]).map_err(|_| EcdsaError::InvalidRValue)?;
        let s = ScalarFE::from_bytes_be(&bytes[32..]).map_err(|_| EcdsaError::InvalidSValue)?;
        Self::new(r, s)
    }
}

/// Sign a message hash using ECDSA.
///
/// # Arguments
/// * `message_hash` - The 32-byte hash of the message to sign
/// * `private_key` - The private key (scalar)
/// * `nonce` - The random nonce k (MUST be cryptographically random and unique per signature)
///
/// # Security Warning
/// This implementation is NOT constant-time. For production use, ensure:
/// 1. The nonce is generated using a CSPRNG
/// 2. Each nonce is used only once
/// 3. Use RFC 6979 for deterministic nonce generation
///
/// # Returns
/// A signature (r, s) or an error if the inputs are invalid.
pub fn sign(
    message_hash: &[u8; 32],
    private_key: &ScalarFE,
    nonce: &ScalarFE,
) -> Result<Signature, EcdsaError> {
    // Validate nonce is not zero
    if *nonce == ScalarFE::zero() {
        return Err(EcdsaError::InvalidNonce);
    }

    // R = k * G
    let generator = Secp256k1Curve::generator();
    let k_canonical = nonce.canonical();
    let r_point = generator.operate_with_self(k_canonical);
    let r_affine = r_point.to_affine();

    // r = R.x mod n
    // Convert x coordinate from base field to scalar field
    let r_x_bytes = r_affine.x().to_bytes_be();
    let r = ScalarFE::from_bytes_be(&r_x_bytes).map_err(|_| EcdsaError::InvalidRValue)?;

    if r == ScalarFE::zero() {
        return Err(EcdsaError::InvalidRValue);
    }

    // z = message_hash as scalar field element
    let z = ScalarFE::from_bytes_be(message_hash).map_err(|_| EcdsaError::InvalidMessageHash)?;

    // k_inv = k^(-1) mod n
    let k_inv = nonce.inv().map_err(|_| EcdsaError::InverseError)?;

    // s = k^(-1) * (z + r * d) mod n
    let mut s = &k_inv * &(&z + &(&r * private_key));

    if s == ScalarFE::zero() {
        return Err(EcdsaError::InvalidSValue);
    }

    // Normalize to low-S form to prevent signature malleability.
    // If s > n/2, use n - s instead.
    if s.canonical() > half_order() {
        s = -s;
    }

    Signature::new(r, s)
}

/// Verify an ECDSA signature.
///
/// # Arguments
/// * `message_hash` - The 32-byte hash of the signed message
/// * `signature` - The signature to verify
/// * `public_key` - The public key point (must be on the curve)
///
/// # Returns
/// `Ok(())` if the signature is valid, `Err` otherwise.
///
/// # Security
/// This function rejects high-S signatures to prevent signature malleability.
pub fn verify(
    message_hash: &[u8; 32],
    signature: &Signature,
    public_key: &CurvePoint,
) -> Result<(), EcdsaError> {
    // Validate r and s are non-zero (they're already reduced mod n by FieldElement)
    if signature.r == ScalarFE::zero() {
        return Err(EcdsaError::InvalidRValue);
    }
    if signature.s == ScalarFE::zero() {
        return Err(EcdsaError::InvalidSValue);
    }

    // Reject high-S signatures to prevent malleability
    if signature.s.canonical() > half_order() {
        return Err(EcdsaError::InvalidSValue);
    }

    // Validate public key is on the curve
    if !is_point_on_curve(public_key) {
        return Err(EcdsaError::InvalidPublicKey);
    }

    // Reject point at infinity as public key
    if *public_key == CurvePoint::neutral_element() {
        return Err(EcdsaError::InvalidPublicKey);
    }

    // z = message_hash as scalar field element
    let z = ScalarFE::from_bytes_be(message_hash).map_err(|_| EcdsaError::InvalidMessageHash)?;

    // s_inv = s^(-1) mod n
    let s_inv = signature.s.inv().map_err(|_| EcdsaError::InverseError)?;

    // u1 = z * s^(-1) mod n
    let u1 = &z * &s_inv;

    // u2 = r * s^(-1) mod n
    let u2 = &signature.r * &s_inv;

    // R' = u1 * G + u2 * Q
    let generator = Secp256k1Curve::generator();
    let u1_g = generator.operate_with_self(u1.canonical());
    let u2_q = public_key.operate_with_self(u2.canonical());
    let r_prime = u1_g.operate_with(&u2_q);

    // Check if R' is the point at infinity
    if r_prime == CurvePoint::neutral_element() {
        return Err(EcdsaError::VerificationFailed);
    }

    let r_prime_affine = r_prime.to_affine();

    // r' = R'.x mod n
    let r_prime_x_bytes = r_prime_affine.x().to_bytes_be();
    let r_prime_scalar =
        ScalarFE::from_bytes_be(&r_prime_x_bytes).map_err(|_| EcdsaError::VerificationFailed)?;

    // Verify r == r'
    if signature.r == r_prime_scalar {
        Ok(())
    } else {
        Err(EcdsaError::VerificationFailed)
    }
}

/// Derive a public key from a private key.
///
/// # Arguments
/// * `private_key` - The private key scalar
///
/// # Returns
/// The corresponding public key point Q = d * G
pub fn derive_public_key(private_key: &ScalarFE) -> CurvePoint {
    let generator = Secp256k1Curve::generator();
    generator.operate_with_self(private_key.canonical())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;

    fn test_private_key() -> ScalarFE {
        // A valid test private key (DO NOT use in production)
        ScalarFE::from_hex_unchecked(
            "c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721",
        )
    }

    fn test_nonce() -> ScalarFE {
        // A valid test nonce (DO NOT reuse in production)
        ScalarFE::from_hex_unchecked(
            "a6e3c57dd01abe90086538398355dd4c3b17aa873382b0f24d6129493d8aad60",
        )
    }

    fn test_message_hash() -> [u8; 32] {
        // SHA256("test message")
        [
            0x3f, 0x0a, 0x37, 0x73, 0x67, 0xf5, 0xa1, 0xb9, 0x5e, 0x33, 0x2d, 0xe8, 0x12, 0x84,
            0x16, 0x6a, 0x68, 0xfb, 0x47, 0x4d, 0x34, 0xd3, 0x18, 0x30, 0x31, 0x48, 0xfd, 0xfd,
            0xd6, 0x58, 0x9c, 0x27,
        ]
    }

    #[test]
    fn test_sign_and_verify() {
        let private_key = test_private_key();
        let public_key = derive_public_key(&private_key);
        let nonce = test_nonce();
        let message_hash = test_message_hash();

        // Sign the message
        let signature = sign(&message_hash, &private_key, &nonce).expect("Signing failed");

        // Verify the signature
        let result = verify(&message_hash, &signature, &public_key);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
    }

    #[test]
    fn test_invalid_signature_fails_verification() {
        let private_key = test_private_key();
        let public_key = derive_public_key(&private_key);
        let nonce = test_nonce();
        let message_hash = test_message_hash();

        // Sign the message
        let signature = sign(&message_hash, &private_key, &nonce).expect("Signing failed");

        // Create an invalid signature by modifying s
        let invalid_signature = Signature::new(signature.r, signature.s + ScalarFE::one()).unwrap();

        // Verification should fail
        let result = verify(&message_hash, &invalid_signature, &public_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_message_fails_verification() {
        let private_key = test_private_key();
        let public_key = derive_public_key(&private_key);
        let nonce = test_nonce();
        let message_hash = test_message_hash();

        // Sign the message
        let signature = sign(&message_hash, &private_key, &nonce).expect("Signing failed");

        // Try to verify with a different message
        let wrong_message = [0u8; 32];
        let result = verify(&wrong_message, &signature, &public_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_public_key_fails_verification() {
        let private_key = test_private_key();
        let nonce = test_nonce();
        let message_hash = test_message_hash();

        // Sign the message
        let signature = sign(&message_hash, &private_key, &nonce).expect("Signing failed");

        // Use a different public key
        let wrong_private_key = ScalarFE::from(12345u64);
        let wrong_public_key = derive_public_key(&wrong_private_key);

        // Verification should fail
        let result = verify(&message_hash, &signature, &wrong_public_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_nonce_rejected() {
        let private_key = test_private_key();
        let nonce = ScalarFE::zero();
        let message_hash = test_message_hash();

        let result = sign(&message_hash, &private_key, &nonce);
        assert_eq!(result, Err(EcdsaError::InvalidNonce));
    }

    #[test]
    fn test_signature_serialization_roundtrip() {
        let private_key = test_private_key();
        let nonce = test_nonce();
        let message_hash = test_message_hash();

        let signature = sign(&message_hash, &private_key, &nonce).expect("Signing failed");

        // Serialize and deserialize
        let bytes = signature.to_bytes();
        let recovered = Signature::from_bytes(&bytes).expect("Deserialization failed");

        assert_eq!(signature, recovered);
    }

    #[test]
    fn test_public_key_derivation() {
        let private_key = test_private_key();
        let public_key = derive_public_key(&private_key);

        // Public key should be on the curve
        let affine = public_key.to_affine();
        let x = affine.x();
        let y = affine.y();

        // Check curve equation: y^2 = x^3 + 7
        let y_sq = y.pow(2u16);
        let x_cubed_plus_b = x.pow(3u16) + Secp256k1Curve::b();
        assert_eq!(y_sq, x_cubed_plus_b);
    }

    #[test]
    fn test_signature_has_low_s() {
        let private_key = test_private_key();
        let nonce = test_nonce();
        let message_hash = test_message_hash();

        let signature = sign(&message_hash, &private_key, &nonce).expect("Signing failed");

        // Verify s is in low-S form (s <= n/2)
        assert!(
            signature.s.canonical() <= super::half_order(),
            "Signature s should be in low-S form"
        );
    }

    #[test]
    fn test_high_s_signature_rejected() {
        let private_key = test_private_key();
        let public_key = derive_public_key(&private_key);
        let nonce = test_nonce();
        let message_hash = test_message_hash();

        let signature = sign(&message_hash, &private_key, &nonce).expect("Signing failed");

        // Create a high-S signature by negating s
        // Note: -s mod n = n - s, which is > n/2 when s < n/2
        let high_s = -signature.s.clone();
        let high_s_signature = Signature {
            r: signature.r,
            s: high_s,
        };

        // High-S signature should be rejected
        let result = verify(&message_hash, &high_s_signature, &public_key);
        assert_eq!(result, Err(EcdsaError::InvalidSValue));
    }

    #[test]
    fn test_point_at_infinity_rejected_as_public_key() {
        let private_key = test_private_key();
        let nonce = test_nonce();
        let message_hash = test_message_hash();

        let signature = sign(&message_hash, &private_key, &nonce).expect("Signing failed");

        // Try to verify with point at infinity as public key
        let infinity = CurvePoint::neutral_element();
        let result = verify(&message_hash, &signature, &infinity);
        assert_eq!(result, Err(EcdsaError::InvalidPublicKey));
    }
}
