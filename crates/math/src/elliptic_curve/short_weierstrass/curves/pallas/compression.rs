//! Point compression for Pallas using SEC1 encoding.
//!
//! Compressed format: 33 bytes (0x02/0x03 prefix + 32 bytes x-coordinate)

use super::curve::PallasCurve;
use crate::{
    elliptic_curve::{
        short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass},
        traits::FromAffine,
    },
    errors::ByteConversionError,
    field::{element::FieldElement, fields::pallas_field::Pallas255PrimeField},
    traits::ByteConversion,
};

use crate::cyclic_group::IsGroup;

type Point = ShortWeierstrassProjectivePoint<PallasCurve>;
type FE = FieldElement<Pallas255PrimeField>;

/// Compress a Pallas point to 33 bytes (SEC1 format).
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

        let y_bytes = y.to_bytes_be();
        let is_odd = y_bytes.last().map(|b| b & 1 == 1).unwrap_or(false);
        result[0] = if is_odd { 0x03 } else { 0x02 };

        result
    }
}

/// Decompress a SEC1-encoded Pallas point from 33 bytes.
pub fn decompress_point(input: &[u8]) -> Result<Point, ByteConversionError> {
    if input.len() != 33 {
        return Err(ByteConversionError::InvalidValue);
    }

    let prefix = input[0];

    if prefix == 0x00 && input[1..].iter().all(|&b| b == 0) {
        return Ok(Point::neutral_element());
    }

    if prefix != 0x02 && prefix != 0x03 {
        return Err(ByteConversionError::InvalidValue);
    }

    let x = FE::from_bytes_be(&input[1..33])?;

    // y² = x³ + 5
    let y_squared = x.pow(3_u16) + PallasCurve::b();

    let (y_sqrt_1, y_sqrt_2) = y_squared.sqrt().ok_or(ByteConversionError::InvalidValue)?;

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
        let g = PallasCurve::generator();
        let compressed = compress_point(&g);
        let decompressed = decompress_point(&compressed).unwrap();
        assert_eq!(g, decompressed);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_2g() {
        let g = PallasCurve::generator();
        let p = g.operate_with_self(UnsignedInteger::<4>::from("2"));
        let compressed = compress_point(&p);
        let decompressed = decompress_point(&compressed).unwrap();
        assert_eq!(p, decompressed);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compress_decompress_large_scalar() {
        let g = PallasCurve::generator();
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
    fn compress_decompress_negated_point() {
        let g = PallasCurve::generator();
        let neg_g = g.neg();
        let compressed = compress_point(&neg_g);
        let decompressed = decompress_point(&compressed).unwrap();
        assert_eq!(neg_g, decompressed);
    }

    #[test]
    fn decompress_invalid_length() {
        assert!(decompress_point(&[0u8; 32]).is_err());
    }

    #[test]
    fn decompress_invalid_prefix() {
        let mut bad = [0u8; 33];
        bad[0] = 0x05;
        assert!(decompress_point(&bad).is_err());
    }
}
