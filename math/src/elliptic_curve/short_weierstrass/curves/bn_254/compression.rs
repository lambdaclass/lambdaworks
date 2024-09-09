use super::{field_extension::BN254PrimeField, twist::BN254TwistCurve};
use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bn_254::{curve::BN254Curve, field_extension::Degree2ExtensionField, sqrt},
        point::ShortWeierstrassProjectivePoint,
        traits::{Compress, IsShortWeierstrass},
    },
    field::element::FieldElement,
};
use core::cmp::Ordering;

use crate::{
    cyclic_group::IsGroup, elliptic_curve::traits::FromAffine, errors::ByteConversionError,
    traits::ByteConversion,
};

type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
type BN254FieldElement = FieldElement<BN254PrimeField>;

/// As we have less than 3 bits available in our coordinate x, we can't follow BLS12-381 style encoding.
/// We use the 2 most significant bits instead
/// 00: uncompressed
/// 10: compressed and y_neg >= y
/// 11: compressed and y_neg < y
/// 01: compressed infinity point
/// the "uncompressed infinity point" will just have 00 (uncompressed) followed by zeroes (infinity = 0,0 in affine coordinates).
/// adapted from gnark https://github.com/consensys/gnark-crypto/blob/v0.13.0/ecc/bn254/marshal.go

impl Compress for BN254Curve {
    type G1Point = G1Point;

    type G2Point = G2Point;

    type G1Compressed = [u8; 32];

    type G2Compressed = [u8; 64];

    type Error = ByteConversionError;

    #[cfg(feature = "alloc")]
    fn compress_g1_point(point: &Self::G1Point) -> Self::G1Compressed {
        if *point == G1Point::neutral_element() {
            // Point is at infinity
            let mut x_bytes = [0_u8; 32];
            x_bytes[0] |= 1 << 6; // x_bytes = 01000000
            x_bytes
        } else {
            // Point is not at infinity
            let point_affine = point.to_affine();
            let x = point_affine.x();
            let y = point_affine.y();

            let mut x_bytes = [0u8; 32];
            let bytes = x.to_bytes_be();
            x_bytes.copy_from_slice(&bytes);
            // Set first bit to 1 to indicate this is a compressed element.
            x_bytes[0] |= 1 << 7; // x_bytes = 10000000

            let y_neg = core::ops::Neg::neg(y);
            if y_neg.representative() < y.representative() {
                x_bytes[0] |= 1 << 6; // x_bytes = 11000000
            }
            x_bytes
        }
    }

    fn decompress_g1_point(input_bytes: &mut [u8]) -> Result<Self::G1Point, Self::Error> {
        // We check that input_bytes has 32 bytes.
        if !input_bytes.len() == 32 {
            return Err(ByteConversionError::InvalidValue);
        }

        let first_byte = input_bytes.first().unwrap();
        // We get the 2 most significant bits
        let prefix_bits = first_byte >> 6;

        // If first two bits are 00, then the value is not compressed.
        if prefix_bits == 0_u8 {
            return Err(ByteConversionError::ValueNotCompressed);
        }

        // If first two bits are 01, then the compressed point is the
        // point at infinity and we return it directly.
        if prefix_bits == 1_u8 {
            return Ok(G1Point::neutral_element());
        }

        let first_byte_without_control_bits = (first_byte << 2) >> 2;
        input_bytes[0] = first_byte_without_control_bits;

        let x = BN254FieldElement::from_bytes_be(input_bytes)?;

        // We apply the elliptic curve formula to know the y^2 value.
        let y_squared = x.pow(3_u16) + BN254FieldElement::from(3);

        let (y_sqrt_1, y_sqrt_2) = &y_squared.sqrt().ok_or(ByteConversionError::InvalidValue)?;

        // If the frist two bits are 10, we take the smaller root.
        // If the first two bits are 11, we take the grater one.
        let y = match (
            y_sqrt_1.representative().cmp(&y_sqrt_2.representative()),
            prefix_bits,
        ) {
            (Ordering::Greater, 2_u8) => y_sqrt_2,
            (Ordering::Greater, _) => y_sqrt_1,
            (Ordering::Less, 2_u8) => y_sqrt_1,
            (Ordering::Less, _) => y_sqrt_2,
            (Ordering::Equal, _) => y_sqrt_1,
        };

        let point =
            G1Point::from_affine(x, y.clone()).map_err(|_| ByteConversionError::InvalidValue)?;

        Ok(point)
    }

    #[cfg(feature = "alloc")]
    fn compress_g2_point(point: &Self::G2Point) -> Self::G2Compressed {
        if *point == G2Point::neutral_element() {
            // Point is at infinity
            let mut x_bytes = [0_u8;64];
            x_bytes[0] |= 1 << 6; // x_bytes = 01000000
            x_bytes
        } else {
            // Point is not at infinity
            let point_affine = point.to_affine();
            let x = point_affine.x();
            let y = point_affine.y();

            let mut x_bytes = [0u8; 64];
            let bytes = x.to_bytes_be();
            x_bytes.copy_from_slice(&bytes);

            // Set first bit to to 1 indicate this is compressed element.
            x_bytes[0] |= 1 << 7;
            let [y0, y1] = y.value();

            // We see if y_neg < y lexicographically where the lexicographic order is as follows:
            // Let a = a0 + a1 * u and b = b0 + b1 * u in Fp2, then a < b if a1 < b1 or
            // a1 = b1 and a0 < b0.
            // TODO: We won't use this prefix in decompress_g2_point. Why?
            if y1 == &BN254FieldElement::zero() {
                let y0_neg = core::ops::Neg::neg(y0);
                if y0_neg.representative() < y0.representative() {
                    x_bytes[0] |= 1 << 6; // Prefix: 11
                }
            } else {
                let y1_neg = core::ops::Neg::neg(y1);
                if y1_neg.representative() < y1.representative() {
                    x_bytes[0] |= 1 << 6; // PRefix: 11
                }
            }

            x_bytes
        }
    }

    #[allow(unused)]
    fn decompress_g2_point(input_bytes: &mut [u8]) -> Result<Self::G2Point, Self::Error> {
        if !input_bytes.len() == 64 {
            return Err(ByteConversionError::InvalidValue);
        }

        let first_byte = input_bytes.first().unwrap();

        // We get the first 2 bits.
        let prefix_bits = first_byte >> 6;

        // If first two bits are 00, then the value is not compressed.
        if prefix_bits == 0_u8 {
            return Err(ByteConversionError::InvalidValue);
        }

        // If the first two bits are 01, then the compressed point is the
        // point at infinity and we return it directly.
        if prefix_bits == 1_u8 {
            return Ok(Self::G2Point::neutral_element());
        }

        let first_byte_without_control_bits = (first_byte << 2) >> 2;
        input_bytes[0] = first_byte_without_control_bits;

        let input0 = &input_bytes[0..32];
        let input1 = &input_bytes[32..];
        let x0 = BN254FieldElement::from_bytes_be(input0).unwrap();
        let x1 = BN254FieldElement::from_bytes_be(input1).unwrap();
        let x: FieldElement<Degree2ExtensionField> = FieldElement::new([x0, x1]);

        let b_param_qfe = BN254TwistCurve::b();
        let mut y = FieldElement::<Degree2ExtensionField>::one();

        //TODO: Why do we always have to set sqrt_qfe input to 0 for tests to pass?

        // If the first two bits are 11, then the square root chosen is the greater one.
        // So we should use sqrt_qfe with the input 1.
        if prefix_bits == 3_u8 {
            y = sqrt::sqrt_qfe(&(x.pow(3_u64) + b_param_qfe), 0)
                .ok_or(ByteConversionError::InvalidValue)?;

        // If the first two bits are 10, then the square root chosen is the smaller one.
        // So we should use sqrt_qfe with the input 0.
        } else {
            y = sqrt::sqrt_qfe(&(x.pow(3_u64) + b_param_qfe), 0)
                .ok_or(ByteConversionError::InvalidValue)?;
        }

        Self::G2Point::from_affine(x, y).map_err(|_| ByteConversionError::InvalidValue)

        //TODO: Do we have to check that the point is in the subgroup?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{BN254FieldElement, G1Point, G2Point};
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::twist::BN254TwistCurve;
    use crate::elliptic_curve::short_weierstrass::traits::Compress;
    use crate::elliptic_curve::traits::{FromAffine, IsEllipticCurve};

    type FpE = BN254FieldElement;
    type Fp2E = FieldElement<Degree2ExtensionField>;

    #[cfg(feature = "alloc")]
    use crate::{
        cyclic_group::IsGroup, traits::ByteConversion, unsigned_integer::element::UnsignedInteger,
    };

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g1_compress_generator() {
        use crate::elliptic_curve::short_weierstrass::traits::Compress;

        let g = BN254Curve::generator();
        let mut compressed_g = BN254Curve::compress_g1_point(&g);
        let first_byte = compressed_g.first().unwrap();

        let first_byte_without_control_bits = (first_byte << 2) >> 2;
        compressed_g[0] = first_byte_without_control_bits;

        let compressed_g_x = BN254FieldElement::from_bytes_be(&compressed_g).unwrap();
        let g_x = g.x();

        assert_eq!(*g_x, compressed_g_x);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g1_compress_point_at_inf() {
        use crate::elliptic_curve::short_weierstrass::traits::Compress;

        let inf = G1Point::neutral_element();
        let compressed_inf = BN254Curve::compress_g1_point(&inf);
        let first_byte = compressed_inf.first().unwrap();

        assert_eq!(*first_byte >> 6, 1_u8);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g1_compress_decompress_is_identity() {
        use crate::elliptic_curve::short_weierstrass::traits::Compress;

        let g = BN254Curve::generator();
        let compressed_g = BN254Curve::compress_g1_point(&g);
        let mut compressed_g_slice: [u8; 32] = compressed_g.try_into().unwrap();

        let decompressed_g = BN254Curve::decompress_g1_point(&mut compressed_g_slice).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g1_compress_decompress_is_identity_2() {
        let g = BN254Curve::generator().operate_with_self(UnsignedInteger::<4>::from("2"));

        let compressed_g = BN254Curve::compress_g1_point(&g);
        let mut compressed_g_slice: [u8; 32] = compressed_g.try_into().unwrap();

        let decompressed_g = BN254Curve::decompress_g1_point(&mut compressed_g_slice).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g2_compress_generator() {
        use crate::elliptic_curve::short_weierstrass::traits::Compress;

        let g = BN254TwistCurve::generator();
        let mut compressed_g = BN254Curve::compress_g2_point(&g);
        let first_byte = compressed_g.first().unwrap();

        let first_byte_without_control_bits = (first_byte << 2) >> 2;
        compressed_g[0] = first_byte_without_control_bits;

        let compressed_g_x =
            FieldElement::<Degree2ExtensionField>::from_bytes_be(&compressed_g).unwrap();
        let g_x = g.x();

        assert_eq!(*g_x, compressed_g_x);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g2_compress_point_at_inf() {
        use crate::elliptic_curve::short_weierstrass::traits::Compress;

        let inf = G2Point::neutral_element();
        let compressed_inf = BN254Curve::compress_g2_point(&inf);
        let first_byte = compressed_inf.first().unwrap();

        assert_eq!(*first_byte >> 6, 1_u8);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g2_compress_decompress_is_identity() {
        use crate::elliptic_curve::short_weierstrass::traits::Compress;

        let g = BN254TwistCurve::generator();
        let compressed_g = BN254Curve::compress_g2_point(&g);
        let mut compressed_g_slice: [u8; 64] = compressed_g.try_into().unwrap();

        let decompressed_g = BN254Curve::decompress_g2_point(&mut compressed_g_slice).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g2_compress_decompress_is_identity_2() {
        let g = BN254TwistCurve::generator().operate_with_self(UnsignedInteger::<4>::from("2"));

        let compressed_g = BN254Curve::compress_g2_point(&g);
        let mut compressed_g_slice: [u8; 64] = compressed_g.try_into().unwrap();

        let decompressed_g = BN254Curve::decompress_g2_point(&mut compressed_g_slice).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g2_compress_decompress_is_identity_3() {
        use crate::unsigned_integer::element::U256;

        let g = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
                )),
            ]),
        )
        .unwrap();

        let compressed_g = BN254Curve::compress_g2_point(&g);
        let mut compressed_g_slice: [u8; 64] = compressed_g.try_into().unwrap();

        let decompressed_g = BN254Curve::decompress_g2_point(&mut compressed_g_slice).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn g2_compress_decompress_is_identity_4() {
        use crate::unsigned_integer::element::U256;

        let g = G2Point::from_affine(
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "3010c68cb50161b7d1d96bb71edfec9880171954e56871abf3d93cc94d745fa1",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "0476be093a6d2b4bbf907172049874af11e1b6267606e00804d3ff0037ec57fd",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "01b33461f39d9e887dbb100f170a2345dde3c07e256d1dfa2b657ba5cd030427",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "14c059d74e5b6c4ec14ae5864ebe23a71781d86c29fb8fb6cce94f70d3de7a21",
                )),
            ]),
        )
        .unwrap();
        // calculate g point operate with itself
        let g_2 = g.operate_with_self(UnsignedInteger::<4>::from("2"));

        let compressed_g2 = BN254Curve::compress_g2_point(&g_2);
        let mut compressed_g2_slice: [u8; 64] = compressed_g2.try_into().unwrap();

        let decompressed_g2 = BN254Curve::decompress_g2_point(&mut compressed_g2_slice).unwrap();

        assert_eq!(g_2, decompressed_g2);
    }

    #[test]
    fn g1_decompress_wrong_bytes_length() {
        let mut input_bytes: [u8; 31] = [0; 31];
        let result = BN254Curve::decompress_g1_point(&mut input_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn g2_decompress_wrong_bytes_length() {
        let mut input_bytes: [u8; 65] = [0; 65];
        let result = BN254Curve::decompress_g2_point(&mut input_bytes);
        assert!(result.is_err());
    }
}
