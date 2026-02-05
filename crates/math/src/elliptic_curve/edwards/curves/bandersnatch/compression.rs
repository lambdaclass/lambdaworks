//! Point compression and decompression for Bandersnatch curve.
//!
//! Edwards curves have a natural compression scheme: store the y-coordinate
//! and a single bit indicating the sign of x. This gives 32-byte compressed
//! points (256 bits for y + 1 bit for x sign).
//!
//! # Compression Format
//!
//! Compressed points are 32 bytes:
//! - Bytes 0-31: y-coordinate in little-endian
//! - The highest bit of the last byte encodes the sign of x
//!
//! # Decompression
//!
//! Given y, we solve for x using the curve equation:
//! ax² + y² = 1 + dx²y²  =>  x² = (1 - y²) / (a - dy²)

use super::curve::{BandersnatchBaseField, BandersnatchCurve};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::edwards::point::EdwardsProjectivePoint;
use crate::elliptic_curve::edwards::traits::IsEdwards;
use crate::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrConfig;
use crate::elliptic_curve::traits::FromAffine;
use crate::errors::ByteConversionError;
use crate::field::element::FieldElement;
use crate::field::fields::montgomery_backed_prime_fields::IsModulus;
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::U256;
use core::cmp::Ordering;

type FE = FieldElement<BandersnatchBaseField>;
type BandersnatchPoint = EdwardsProjectivePoint<BandersnatchCurve>;

/// Size of a compressed Bandersnatch point in bytes.
pub const COMPRESSED_POINT_SIZE: usize = 32;

/// Compress a Bandersnatch point to 32 bytes.
///
/// Stores y-coordinate with x's sign bit in the highest bit of the last byte.
#[cfg(feature = "alloc")]
pub fn compress(point: &BandersnatchPoint) -> [u8; COMPRESSED_POINT_SIZE] {
    if point.is_neutral_element() {
        let mut result = [0u8; COMPRESSED_POINT_SIZE];
        result[0] = 1; // y = 1 in little-endian
        return result;
    }

    let affine = point.to_affine();
    let x = affine.x();
    let y = affine.y();

    let mut result: [u8; 32] = y.to_bytes_le().try_into().expect("y is 32 bytes");

    // Set sign bit if x is "negative" (lexicographically larger)
    let x_neg = -x;
    if x.canonical().cmp(&x_neg.canonical()) == Ordering::Greater {
        result[COMPRESSED_POINT_SIZE - 1] |= 0x80;
    }

    result
}

/// Decompress a 32-byte compressed point.
pub fn decompress(
    bytes: &[u8; COMPRESSED_POINT_SIZE],
) -> Result<BandersnatchPoint, ByteConversionError> {
    let mut y_bytes = *bytes;

    let sign_bit = (y_bytes[COMPRESSED_POINT_SIZE - 1] & 0x80) != 0;
    y_bytes[COMPRESSED_POINT_SIZE - 1] &= 0x7F;

    let y_uint = U256::from_bytes_le(&y_bytes).map_err(|_| ByteConversionError::InvalidValue)?;

    if y_uint >= FrConfig::MODULUS {
        return Err(ByteConversionError::InvalidValue);
    }

    let y = FE::new(y_uint);

    if y == FE::one() && !sign_bit {
        return Ok(BandersnatchPoint::neutral_element());
    }

    // x² = (1 - y²) / (a - dy²)
    let a = BandersnatchCurve::a();
    let d = BandersnatchCurve::d();
    let y_squared = y.square();
    let numerator = FE::one() - &y_squared;
    let denominator = &a - &d * &y_squared;

    if denominator == FE::zero() {
        return Err(ByteConversionError::InvalidValue);
    }

    let x_squared = (&numerator / &denominator).map_err(|_| ByteConversionError::InvalidValue)?;
    let (x1, x2) = x_squared.sqrt().ok_or(ByteConversionError::InvalidValue)?;

    let x = match (x1.canonical().cmp(&x2.canonical()), sign_bit) {
        (Ordering::Greater, true) => x1,
        (Ordering::Greater, false) => x2,
        (Ordering::Less, true) => x2,
        (Ordering::Less, false) => x1,
        (Ordering::Equal, _) => x1,
    };

    BandersnatchPoint::from_affine(x, y).map_err(|_| ByteConversionError::InvalidValue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elliptic_curve::traits::IsEllipticCurve;

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_roundtrip() {
        let g = BandersnatchCurve::generator();
        let compressed = compress(&g);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(g.to_affine(), decompressed.to_affine());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_neutral() {
        let neutral = BandersnatchPoint::neutral_element();
        let compressed = compress(&neutral);
        assert!(decompress(&compressed).unwrap().is_neutral_element());
    }

    #[test]
    fn invalid_y_rejected() {
        let mut bytes = [0xFFu8; COMPRESSED_POINT_SIZE];
        bytes[COMPRESSED_POINT_SIZE - 1] &= 0x7F;
        assert!(decompress(&bytes).is_err());
    }
}
