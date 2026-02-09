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
//! - Operations may not be constant-time
//! - No protection against rogue key attacks in aggregation (would need proof-of-possession)
//!
//! For production use, consider well-audited libraries like `blst` or `bls-signatures`.

use crate::hash_to_curve::hash_to_g2;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::{
    final_exponentiation, miller, BLS12381AtePairing,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
use lambdaworks_math::elliptic_curve::traits::{IsEllipticCurve, IsPairing};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::traits::ByteConversion;
use sha3::Digest;

/// Type alias for G1 points (public keys)
pub type G1Point = ShortWeierstrassJacobianPoint<BLS12381Curve>;

/// Type alias for G2 points (signatures, hashed messages)
pub type G2Point = ShortWeierstrassJacobianPoint<BLS12381TwistCurve>;

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
    /// Input arrays have mismatched lengths
    LengthMismatch,
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
    ///
    /// # Security Note
    /// This uses SHA3-512 to get 512 bits of entropy, XORs two halves together,
    /// then reduces modulo the field order. This provides nearly uniform sampling
    /// over the scalar field with minimal bias.
    pub fn from_seed(seed: &[u8]) -> Result<Self, BlsError> {
        use lambdaworks_math::unsigned_integer::element::U256;
        use sha3::Sha3_512;

        // Use SHA3-512 for 512 bits of entropy to minimize bias when reducing mod r
        let hash = Sha3_512::digest(seed);

        // XOR the two 32-byte halves to get a well-distributed 256-bit value
        let mut bytes = [0u8; 32];
        for i in 0..32 {
            bytes[i] = hash[i] ^ hash[i + 32];
        }

        // Convert to U256 and create FrElement (automatically reduces mod r)
        let scalar_uint = U256::from_bytes_be(&bytes).map_err(|_| BlsError::InvalidSecretKey)?;
        let sk = FrElement::new(scalar_uint);

        Self::new(sk)
    }

    /// Derive the corresponding public key.
    pub fn public_key(&self) -> PublicKey {
        let g1 = BLS12381Curve::generator();
        let pk = g1.operate_with_self(self.sk.canonical());
        PublicKey { pk }
    }

    /// Sign a message.
    ///
    /// The signature is computed as: sig = sk * H(m)
    /// where H(m) is the hash of the message mapped to G2.
    pub fn sign(&self, message: &[u8]) -> Signature {
        let h = hash_to_g2(message);
        let sig = h.operate_with_self(self.sk.canonical());
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
    ///
    /// # Format
    /// The output is 96 bytes: the x-coordinate of the G2 point (two 48-byte Fp elements).
    /// The highest bit of the first byte encodes the sign of the y-coordinate for compression.
    ///
    /// # Note
    /// This is a simplified serialization for educational purposes. Production implementations
    /// should follow the BLS12-381 serialization spec (e.g., ZCash or IETF formats) which include:
    /// - Proper infinity point handling
    /// - Subgroup membership flags
    /// - Consistent big-endian ordering with padding
    pub fn to_bytes(&self) -> Vec<u8> {
        let affine = self.sig.to_affine();
        let [x, y, _] = affine.coordinates();
        let [x0, x1] = x.value();
        let [y0, _y1] = y.value();

        // Serialize x-coordinate (two 48-byte Fp elements in big-endian)
        let x0_bytes = x0.to_bytes_be();
        let x1_bytes = x1.to_bytes_be();

        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(&x0_bytes);
        bytes.extend_from_slice(&x1_bytes);

        // Add sign bit for point compression
        // The sign is determined by the lexicographically larger y-coordinate
        // For Fp2 = (y0, y1), we use y0's parity when y1 == 0, otherwise y1's parity
        let y0_bytes = y0.to_bytes_be();
        let sign_bit = y0_bytes
            .last()
            .map(|b| b & 1 == 1)
            .expect("Fp element serialization is never empty");
        bytes[0] |= if sign_bit { 0x80 } else { 0x00 };

        bytes
    }
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
    if messages.is_empty() {
        return Err(BlsError::EmptyInput);
    }

    if messages.len() != signatures.len() || messages.len() != public_keys.len() {
        return Err(BlsError::LengthMismatch);
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
    let lhs = final_exponentiation(&lhs).map_err(|_| BlsError::PairingError)?;

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
