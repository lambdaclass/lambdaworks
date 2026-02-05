//! Point compression and decompression for Bandersnatch curve.
//!
//! Edwards curves have a natural compression scheme: store the y-coordinate
//! and a single bit indicating the sign of x. This gives 32-byte compressed
//! points (256 bits for y + 1 bit for x sign).
//!
//! # Compression Format
//!
//! Compressed points are 32 bytes (256 bits):
//! - Bytes 0-31: y-coordinate in little-endian
//! - The highest bit of the last byte encodes the sign of x
//!
//! # Decompression
//!
//! Given y, we solve for x using the curve equation:
//! ax² + y² = 1 + dx²y²
//!
//! Rearranging: x² = (1 - y²) / (a - dy²)

use super::curve::{BandersnatchCurve, BaseBandersnatchFieldElement};
use super::field::BANDERSNATCH_PRIME_FIELD_ORDER;
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::edwards::point::EdwardsProjectivePoint;
use crate::elliptic_curve::edwards::traits::IsEdwards;
use crate::elliptic_curve::traits::FromAffine;
use crate::errors::ByteConversionError;
use crate::field::element::FieldElement;
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::U256;
use core::cmp::Ordering;

type FE = FieldElement<BaseBandersnatchFieldElement>;
type BandersnatchPoint = EdwardsProjectivePoint<BandersnatchCurve>;

/// Size of a compressed Bandersnatch point in bytes (32 bytes).
pub const COMPRESSED_POINT_SIZE: usize = 32;

/// Compress a Bandersnatch point to 32 bytes.
///
/// The compression format stores:
/// - The y-coordinate in little-endian (32 bytes)
/// - The sign bit of x in the highest bit of the last byte
///
/// # Arguments
///
/// * `point` - The point to compress
///
/// # Returns
///
/// A 32-byte array containing the compressed point
#[cfg(feature = "alloc")]
pub fn compress(point: &BandersnatchPoint) -> [u8; COMPRESSED_POINT_SIZE] {
    if point.is_neutral_element() {
        // Neutral element (0, 1) - y = 1
        let mut result = [0u8; COMPRESSED_POINT_SIZE];
        result[0] = 1; // y = 1 in little-endian
        return result;
    }

    let affine = point.to_affine();
    let x = affine.x();
    let y = affine.y();

    // Serialize y in little-endian
    let mut result = y.to_bytes_le();

    // Determine the sign of x
    // We define "negative" as the lexicographically larger representation
    let x_neg = -x;
    let is_x_negative = x.canonical().cmp(&x_neg.canonical()) == Ordering::Greater;

    // Set the sign bit in the highest bit of the last byte
    if is_x_negative {
        result[COMPRESSED_POINT_SIZE - 1] |= 0x80;
    }

    result.try_into().expect("slice length is 32")
}

/// Decompress a 32-byte compressed point to a Bandersnatch point.
///
/// # Arguments
///
/// * `bytes` - A 32-byte slice containing the compressed point
///
/// # Returns
///
/// The decompressed point, or an error if decompression fails
///
/// # Errors
///
/// Returns `ByteConversionError::InvalidValue` if:
/// - The y-coordinate is not a valid field element
/// - No valid x-coordinate exists for the given y
pub fn decompress(
    bytes: &[u8; COMPRESSED_POINT_SIZE],
) -> Result<BandersnatchPoint, ByteConversionError> {
    let mut y_bytes = *bytes;

    // Extract and clear the sign bit
    let sign_bit = (y_bytes[COMPRESSED_POINT_SIZE - 1] & 0x80) != 0;
    y_bytes[COMPRESSED_POINT_SIZE - 1] &= 0x7F;

    // Parse y-coordinate (little-endian)
    let y_uint = U256::from_bytes_le(&y_bytes).map_err(|_| ByteConversionError::InvalidValue)?;

    // Check if y is within the field
    if y_uint >= BANDERSNATCH_PRIME_FIELD_ORDER {
        return Err(ByteConversionError::InvalidValue);
    }

    let y = FE::new(y_uint);

    // Handle the neutral element case (y = 1)
    if y == FE::one() && !sign_bit {
        return Ok(BandersnatchPoint::neutral_element());
    }

    // Compute x² from the curve equation: ax² + y² = 1 + dx²y²
    // Rearranging: x²(a - dy²) = 1 - y²
    // Therefore: x² = (1 - y²) / (a - dy²)
    let a = BandersnatchCurve::a();
    let d = BandersnatchCurve::d();

    let y_squared = y.square();
    let numerator = FE::one() - &y_squared;
    let denominator = &a - &d * &y_squared;

    // Check if denominator is zero (should not happen for valid points)
    if denominator == FE::zero() {
        return Err(ByteConversionError::InvalidValue);
    }

    let x_squared = (&numerator / &denominator).map_err(|_| ByteConversionError::InvalidValue)?;

    // Compute sqrt(x²)
    let (x1, x2) = x_squared.sqrt().ok_or(ByteConversionError::InvalidValue)?;

    // Choose the correct root based on the sign bit
    let x = match (x1.canonical().cmp(&x2.canonical()), sign_bit) {
        // x1 is "larger" (negative), x2 is "smaller" (positive)
        (Ordering::Greater, true) => x1,  // want negative
        (Ordering::Greater, false) => x2, // want positive
        (Ordering::Less, true) => x2,     // x2 is larger, want negative
        (Ordering::Less, false) => x1,    // x1 is smaller, want positive
        (Ordering::Equal, _) => x1,       // both equal, doesn't matter
    };

    BandersnatchPoint::from_affine(x, y).map_err(|_| ByteConversionError::InvalidValue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elliptic_curve::traits::IsEllipticCurve;

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_generator() {
        let g = BandersnatchCurve::generator();
        let compressed = compress(&g);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(g.to_affine(), decompressed.to_affine());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_neutral_element() {
        let neutral = BandersnatchPoint::neutral_element();
        let compressed = compress(&neutral);
        let decompressed = decompress(&compressed).unwrap();
        assert!(decompressed.is_neutral_element());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_arbitrary_points() {
        let g = BandersnatchCurve::generator();

        // Test several scalar multiples
        for k in [2u64, 5, 100, 12345] {
            let p = g.operate_with_self(k);
            let compressed = compress(&p);
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(p.to_affine(), decompressed.to_affine());
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_negated_point() {
        let g = BandersnatchCurve::generator();
        let neg_g = g.neg();

        let compressed = compress(&neg_g);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(neg_g.to_affine(), decompressed.to_affine());
    }

    #[test]
    fn decompress_invalid_y_fails() {
        // A y value larger than the field modulus should fail
        let mut bytes = [0xFFu8; COMPRESSED_POINT_SIZE];
        bytes[COMPRESSED_POINT_SIZE - 1] &= 0x7F; // Clear sign bit

        let result = decompress(&bytes);
        assert!(result.is_err());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compressed_size_is_32_bytes() {
        let g = BandersnatchCurve::generator();
        let compressed = compress(&g);
        assert_eq!(compressed.len(), 32);
    }
}
