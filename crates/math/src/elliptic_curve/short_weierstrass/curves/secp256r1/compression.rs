//! Point compression for secp256r1 (P-256/NIST P-256) using SEC1 encoding.
//!
//! Compressed format: 33 bytes
//! - Byte 0: 0x02 (y is even) or 0x03 (y is odd)
//! - Bytes 1..33: x-coordinate in big-endian
//!
//! secp256r1 has a 256-bit field, so no spare bits in 32 bytes — requires a prefix byte.

use super::curve::Secp256r1Curve;
use crate::{
    elliptic_curve::{
        short_weierstrass::{
            point::ShortWeierstrassProjectivePoint,
            traits::IsShortWeierstrass,
        },
        traits::FromAffine,
    },
    errors::ByteConversionError,
    field::{element::FieldElement, fields::secp256r1_field::Secp256r1PrimeField},
    traits::ByteConversion,
};

use crate::cyclic_group::IsGroup;

type Point = ShortWeierstrassProjectivePoint<Secp256r1Curve>;
type FE = FieldElement<Secp256r1PrimeField>;

/// Compress a secp256r1 point to 33 bytes (SEC1 format).
///
/// Returns `[0x00; 33]` for the point at infinity (non-standard but unambiguous).
#[cfg(feature = "alloc")]
pub fn compress_point(point: &Point) -> [u8; 33] {
    if *point == Point::neutral_element() {
        [0u8; 33]
    } else {
        let point_affine = point.to_affine();
        let x = point_affine.x();
        let y = point_affine.y();

        let x_bytes = x.to_bytes_be();
        let mut result = [0u8; 33];
        result[1..33].copy_from_slice(&x_bytes);

        // SEC1: 0x02 if y is even, 0x03 if y is odd
        let y_bytes = y.to_bytes_be();
        let is_odd = y_bytes.last().map(|b| b & 1 == 1).unwrap_or(false);
        result[0] = if is_odd { 0x03 } else { 0x02 };

        result
    }
}

/// Decompress a SEC1-encoded secp256r1 point from 33 bytes.
pub fn decompress_point(input: &[u8]) -> Result<Point, ByteConversionError> {
    if input.len() != 33 {
        return Err(ByteConversionError::InvalidValue);
    }

    let prefix = input[0];

    // All zeros = point at infinity
    if prefix == 0x00 && input[1..].iter().all(|&b| b == 0) {
        return Ok(Point::neutral_element());
    }

    if prefix != 0x02 && prefix != 0x03 {
        return Err(ByteConversionError::InvalidValue);
    }

    let x = FE::from_bytes_be(&input[1..33])?;

    // y² = x³ + ax + b
    let y_squared = x.pow(3_u16) + Secp256r1Curve::a() * &x + Secp256r1Curve::b();

    let (y_sqrt_1, y_sqrt_2) = y_squared.sqrt().ok_or(ByteConversionError::InvalidValue)?;

    // Pick the root matching the parity indicated by the prefix
    let want_odd = prefix == 0x03;
    let y1_bytes = y_sqrt_1.to_bytes_be();
    let y1_is_odd = y1_bytes.last().map(|b| b & 1 == 1).unwrap_or(false);

    let y = if y1_is_odd == want_odd {
        y_sqrt_1
    } else {
        y_sqrt_2
    };

    Point::from_affine(x, y).map_err(|_| ByteConversionError::InvalidValue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve};

    #[cfg(feature = "alloc")]
    use crate::unsigned_integer::element::UnsignedInteger;

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_generator() {
        let g = Secp256r1Curve::generator();
        let compressed = compress_point(&g);
        let decompressed = decompress_point(&compressed).unwrap();
        assert_eq!(g, decompressed);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_2g() {
        let g = Secp256r1Curve::generator();
        let p = g.operate_with_self(UnsignedInteger::<4>::from("2"));
        let compressed = compress_point(&p);
        let decompressed = decompress_point(&compressed).unwrap();
        assert_eq!(p, decompressed);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_large_scalar() {
        let g = Secp256r1Curve::generator();
        let p = g.operate_with_self(123456789u64);
        let compressed = compress_point(&p);
        let decompressed = decompress_point(&compressed).unwrap();
        assert_eq!(p, decompressed);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_infinity() {
        let inf = Point::neutral_element();
        let compressed = compress_point(&inf);
        let decompressed = decompress_point(&compressed).unwrap();
        assert_eq!(inf, decompressed);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compressed_prefix_is_02_or_03() {
        let g = Secp256r1Curve::generator();
        let compressed = compress_point(&g);
        assert!(compressed[0] == 0x02 || compressed[0] == 0x03);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_negated_point() {
        let g = Secp256r1Curve::generator();
        let neg_g = g.neg();
        let compressed = compress_point(&neg_g);
        let decompressed = decompress_point(&compressed).unwrap();
        assert_eq!(neg_g, decompressed);
    }

    #[test]
    fn decompress_invalid_length() {
        let bad = [0u8; 32];
        assert!(decompress_point(&bad).is_err());
    }

    #[test]
    fn decompress_invalid_prefix() {
        let mut bad = [0u8; 33];
        bad[0] = 0x05;
        assert!(decompress_point(&bad).is_err());
    }
}
