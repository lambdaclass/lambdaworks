//! Point compression for Pallas using SEC1 encoding.
//!
//! Compressed format: 33 bytes (0x02/0x03 prefix + 32 bytes x-coordinate)

use super::curve::PallasCurve;
use crate::{
    elliptic_curve::short_weierstrass::{
        compression::decompress_sec1, point::ShortWeierstrassProjectivePoint,
    },
    errors::ByteConversionError,
};

#[cfg(feature = "alloc")]
use crate::elliptic_curve::short_weierstrass::compression::compress_sec1;

type Point = ShortWeierstrassProjectivePoint<PallasCurve>;

#[cfg(feature = "alloc")]
pub fn compress_point(point: &Point) -> alloc::vec::Vec<u8> {
    compress_sec1(point)
}

pub fn decompress_point(input: &[u8]) -> Result<Point, ByteConversionError> {
    decompress_sec1(input)
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
