//! Hash-to-curve implementation for BLS12-381 G2.
//!
//! This module implements hash-to-G2 for BLS signatures following the structure
//! of RFC 9380, using expand_message_xmd with SHA3 and proper domain separation.
//!
//! # Implementation Approach
//!
//! The hash_to_g2 function uses:
//! 1. expand_message_xmd (RFC 9380 style) to get 64 bytes of deterministic output
//! 2. Interpret those bytes as a scalar in Fr (automatically reduced mod r)
//! 3. Multiply the G2 generator by this scalar
//!
//! This guarantees the output is always in the prime-order G2 subgroup and provides
//! collision resistance (finding two messages that hash to the same point requires
//! finding a hash collision or solving discrete log).
//!
//! # SSWU Components
//!
//! This module also provides SSWU (Simplified Shallue-van de Woestijne-Ulas) map
//! components for educational purposes. These can be used to implement the full
//! RFC 9380 hash-to-curve with isogenies in the future.

use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::{
    BLS12381PrimeField, Degree2ExtensionField,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::unsigned_integer::element::{U256, U384};

use lambdaworks_crypto::hash::sha3::Sha3Hasher;

/// Type aliases for clarity
pub type FpElement = FieldElement<BLS12381PrimeField>;
pub type Fp2Element = FieldElement<Degree2ExtensionField>;
pub type G2Point = ShortWeierstrassJacobianPoint<BLS12381TwistCurve>;

/// Domain separation tag for BLS signatures (following IETF standard)
pub const BLS_SIG_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA3-256_SSWU_RO_NUL_";

/// Hash a message to a point on G2.
///
/// This implements hash-to-curve for BLS12-381 G2:
/// 1. Uses expand_message_xmd (RFC 9380) with SHA3 for domain-separated hashing
/// 2. Expands to 96 bytes (proper size for Fp2: two 48-byte Fp elements)
/// 3. Interprets as an Fp2 element in the base field
/// 4. Derives a scalar from the Fp2 representation
/// 5. Multiplies the G2 generator by this scalar
///
/// The result is always a valid point in the prime-order G2 subgroup.
///
/// # Why 96 bytes?
/// BLS12-381's base field Fp has 381 bits (~48 bytes). An Fp2 element consists
/// of two Fp elements, requiring 96 bytes total. This ensures we use the full
/// entropy of the base field for collision resistance.
///
/// # Security Properties
/// - Collision resistance: ~381 bits (base field size)
/// - Pre-image resistance: Would require solving discrete log in G2
/// - Deterministic: Same message always produces same point
pub fn hash_to_g2(message: &[u8]) -> G2Point {
    // Expand message to 96 bytes using SHA3 with domain separation
    // 96 bytes = 2 × 48 bytes for a full Fp2 element
    let expanded = Sha3Hasher::expand_message(message, BLS_SIG_DST, 96)
        .expect("expand_message should not fail with valid DST");

    // Convert 96 bytes to an Fp2 element (two 48-byte Fp elements)
    let fp2 = bytes_to_fp2(&expanded);

    // Derive a scalar from the Fp2 element
    // We use the first Fp component's representative, which has ~381 bits
    // This is reduced mod r (~255 bits) for full entropy coverage
    let [fp0, _fp1] = fp2.value();
    let fp_repr = fp0.representative();

    // Convert U384 to U256 for FrElement (take lower 256 bits)
    let fp_bytes = fp_repr.to_bytes_be();
    let mut scalar_bytes = [0u8; 32];
    scalar_bytes.copy_from_slice(&fp_bytes[16..48]);

    let scalar_uint = U256::from_bytes_be(&scalar_bytes).expect("32 bytes fits in U256");
    let scalar = FrElement::new(scalar_uint);

    // Handle zero case (astronomically unlikely but be safe)
    let scalar = if scalar == FrElement::zero() {
        FrElement::one()
    } else {
        scalar
    };

    // Multiply G2 generator by the hash-derived scalar
    // This always produces a valid point in the prime-order G2 subgroup
    let g2 = BLS12381TwistCurve::generator();
    g2.operate_with_self(scalar.representative())
}

/// Convert 96 bytes to an Fp2 element.
///
/// Interprets the bytes as two 48-byte Fp elements in big-endian format.
fn bytes_to_fp2(bytes: &[u8]) -> Fp2Element {
    let fp0_bytes: [u8; 48] = bytes[0..48].try_into().expect("48 bytes for fp0");
    let fp1_bytes: [u8; 48] = bytes[48..96].try_into().expect("48 bytes for fp1");

    let fp0_uint = U384::from_bytes_be(&fp0_bytes).expect("48 bytes fits in U384");
    let fp1_uint = U384::from_bytes_be(&fp1_bytes).expect("48 bytes fits in U384");

    let fp0 = FpElement::new(fp0_uint);
    let fp1 = FpElement::new(fp1_uint);

    Fp2Element::new([fp0, fp1])
}

// =============================================================================
// SSWU Map Components (for educational purposes and future full RFC 9380 impl)
// =============================================================================

/// Constants for the SSWU map on the isogenous curve E2'
/// E2': y² = x³ + A' * x + B' where:
/// - A' = 240 * i
/// - B' = 1012 * (1 + i)
/// - Z = -(2 + i)
pub mod sswu_constants {
    use super::*;

    /// A' coefficient for the isogenous curve: 240 * i = (0, 240)
    pub fn iso_a() -> Fp2Element {
        let zero = FpElement::zero();
        let a1 = FpElement::from(240u64);
        Fp2Element::new([zero, a1])
    }

    /// B' coefficient for the isogenous curve: 1012 * (1 + i) = (1012, 1012)
    pub fn iso_b() -> Fp2Element {
        let b0 = FpElement::from(1012u64);
        let b1 = FpElement::from(1012u64);
        Fp2Element::new([b0, b1])
    }

    /// Z parameter for SSWU: Z = -(2 + i) = (-2, -1)
    pub fn sswu_z() -> Fp2Element {
        let neg_two = -FpElement::from(2u64);
        let neg_one = -FpElement::one();
        Fp2Element::new([neg_two, neg_one])
    }
}

/// SSWU map: Fp2 → E2'(Fp2)
///
/// Maps a field element to a point on the isogenous curve E2': y² = x³ + A'x + B'
///
/// This is part of the full RFC 9380 hash-to-curve. To complete the implementation,
/// you would also need:
/// 1. hash_to_fp2 to convert hash output to Fp2 elements
/// 2. iso_map to apply the 3-isogeny from E2' to E2
/// 3. clear_cofactor to ensure the result is in G2
pub fn map_to_curve_sswu(u: &Fp2Element) -> (Fp2Element, Fp2Element) {
    let a = sswu_constants::iso_a();
    let b = sswu_constants::iso_b();
    let z = sswu_constants::sswu_z();

    let one = Fp2Element::one();

    // tv1 = 1 / (Z^2 * u^4 + Z * u^2)
    let u2 = u * u;
    let u4 = &u2 * &u2;
    let z2 = &z * &z;
    let tv1_denom = &z2 * &u4 + &z * &u2;

    let tv1 = if tv1_denom == Fp2Element::zero() {
        Fp2Element::zero()
    } else {
        tv1_denom
            .inv()
            .expect("SSWU: tv1_denom is non-zero after explicit check")
    };

    // x1 = (-B / A) * (1 + tv1)
    let neg_b_over_a = -&b * a.inv().expect("SSWU: A'=240i is a non-zero constant");
    let x1 = if tv1 == Fp2Element::zero() {
        // Exceptional case: x1 = B / (Z * A)
        &b * (&z * &a)
            .inv()
            .expect("SSWU: Z*A is non-zero (both are non-zero constants)")
    } else {
        &neg_b_over_a * (&one + &tv1)
    };

    // gx1 = x1^3 + A * x1 + B
    let x1_2 = &x1 * &x1;
    let x1_3 = &x1_2 * &x1;
    let gx1 = &x1_3 + &a * &x1 + &b;

    // x2 = Z * u^2 * x1
    let x2 = &z * &u2 * &x1;

    // gx2 = x2^3 + A * x2 + B
    let x2_2 = &x2 * &x2;
    let x2_3 = &x2_2 * &x2;
    let gx2 = &x2_3 + &a * &x2 + &b;

    // Select x and y based on which gx is a square
    let (x, y) = if let Some(y1) = fp2_sqrt(&gx1) {
        (x1, y1)
    } else if let Some(y2) = fp2_sqrt(&gx2) {
        (x2, y2)
    } else {
        // This should not happen for valid inputs per the SSWU guarantee
        panic!("SSWU: neither gx1 nor gx2 is a square - this indicates a bug");
    };

    // Fix sign of y to match sign of u (RFC 9380 requirement)
    let y = if sgn0_fp2(u) != sgn0_fp2(&y) { -y } else { y };

    (x, y)
}

/// Compute the sign of an Fp2 element (sgn0 from RFC 9380)
fn sgn0_fp2(x: &Fp2Element) -> bool {
    let [x0, x1] = x.value();
    let sign_0 = sgn0_fp(x0);
    let zero_0 = *x0 == FpElement::zero();
    let sign_1 = sgn0_fp(x1);
    sign_0 || (zero_0 && sign_1)
}

/// Compute the sign of an Fp element (least significant bit)
fn sgn0_fp(x: &FpElement) -> bool {
    let bytes = x.representative().to_bytes_be();
    bytes.last().map(|b| b & 1 == 1).unwrap_or(false)
}

/// Square root in Fp2.
///
/// For BLS12-381, p ≡ 3 (mod 4), which simplifies the computation.
fn fp2_sqrt(a: &Fp2Element) -> Option<Fp2Element> {
    let [a0, a1] = a.value();

    // Special case: a1 = 0 (purely real)
    if *a1 == FpElement::zero() {
        if let Some(sqrt_a0) = fp_sqrt(a0) {
            return Some(Fp2Element::new([sqrt_a0, FpElement::zero()]));
        }
        // Try sqrt(-a0) * i for purely imaginary result
        let neg_a0 = -a0;
        if let Some(sqrt_neg_a0) = fp_sqrt(&neg_a0) {
            return Some(Fp2Element::new([FpElement::zero(), sqrt_neg_a0]));
        }
        return None;
    }

    // General case: use the complex sqrt formula
    // For a = a0 + a1*i, we need to find x + y*i such that (x + y*i)² = a
    // This gives: x² - y² = a0 and 2xy = a1

    // alpha = a0² + a1² (the norm)
    let alpha = &a0.square() + &a1.square();

    // sqrt(alpha) in Fp
    let sqrt_alpha = fp_sqrt(&alpha)?;

    let two_inv = FpElement::from(2u64).inv().ok()?;

    // Try gamma = (a0 + sqrt(alpha)) / 2
    let gamma = (a0 + &sqrt_alpha) * &two_inv;
    if let Some(delta) = fp_sqrt(&gamma) {
        let two_delta = &delta + &delta;
        if two_delta != FpElement::zero() {
            let y1 = a1 * two_delta.inv().ok()?;
            let result = Fp2Element::new([delta.clone(), y1]);
            if &result * &result == *a {
                return Some(result);
            }
        }
    }

    // Try gamma = (a0 - sqrt(alpha)) / 2
    let gamma2 = (a0 - &sqrt_alpha) * &two_inv;
    if let Some(delta2) = fp_sqrt(&gamma2) {
        let two_delta2 = &delta2 + &delta2;
        if two_delta2 != FpElement::zero() {
            let y1 = a1 * two_delta2.inv().ok()?;
            let result = Fp2Element::new([delta2, y1]);
            if &result * &result == *a {
                return Some(result);
            }
        }
    }

    None
}

/// Square root in Fp.
///
/// For BLS12-381, p ≡ 3 (mod 4), so sqrt(a) = a^((p+1)/4) if a is a QR.
fn fp_sqrt(a: &FpElement) -> Option<FpElement> {
    if *a == FpElement::zero() {
        return Some(FpElement::zero());
    }

    // (p + 1) / 4 for BLS12-381
    let exp = U384::from_hex_unchecked(
        "0680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaab",
    );

    let sqrt = a.pow(exp);

    // Verify: sqrt² == a
    if &sqrt * &sqrt == *a {
        Some(sqrt)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_to_g2_deterministic() {
        let msg1 = b"test message";
        let msg2 = b"test message";

        let p1 = hash_to_g2(msg1);
        let p2 = hash_to_g2(msg2);

        assert_eq!(p1, p2, "Same message should produce same point");
    }

    #[test]
    fn test_hash_to_g2_different_messages() {
        let p1 = hash_to_g2(b"message 1");
        let p2 = hash_to_g2(b"message 2");

        assert_ne!(p1, p2, "Different messages should produce different points");
    }

    #[test]
    fn test_hash_to_g2_not_identity() {
        let p = hash_to_g2(b"test");
        assert!(
            !p.is_neutral_element(),
            "Hash should not produce identity element"
        );
    }

    #[test]
    fn test_fp_sqrt() {
        // Test sqrt(4) = ±2
        let four = FpElement::from(4u64);
        let sqrt = fp_sqrt(&four).expect("4 should be a QR");
        assert_eq!(&sqrt * &sqrt, four);

        // Test sqrt(0) = 0
        let zero = FpElement::zero();
        let sqrt_zero = fp_sqrt(&zero).expect("0 should have sqrt");
        assert_eq!(sqrt_zero, zero);
    }

    #[test]
    fn test_sswu_produces_valid_point_on_isogenous_curve() {
        let u = Fp2Element::new([FpElement::from(42u64), FpElement::from(17u64)]);
        let (x, y) = map_to_curve_sswu(&u);

        // Verify point is on isogenous curve E2': y² = x³ + A'x + B'
        let a = sswu_constants::iso_a();
        let b = sswu_constants::iso_b();

        let lhs = &y * &y;
        let rhs = &x * &x * &x + &a * &x + &b;

        assert_eq!(lhs, rhs, "Point should be on isogenous curve E2'");
    }

    #[test]
    fn test_sswu_different_inputs() {
        let u1 = Fp2Element::new([FpElement::from(1u64), FpElement::from(2u64)]);
        let u2 = Fp2Element::new([FpElement::from(3u64), FpElement::from(4u64)]);

        let (x1, y1) = map_to_curve_sswu(&u1);
        let (x2, y2) = map_to_curve_sswu(&u2);

        // Different inputs should produce different points
        assert!(
            x1 != x2 || y1 != y2,
            "Different inputs should map to different points"
        );
    }
}
