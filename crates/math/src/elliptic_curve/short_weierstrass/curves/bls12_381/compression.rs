use super::{field_extension::BLS12381PrimeField, twist::BLS12381TwistCurve};
use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bls12_381::{curve::BLS12381Curve, field_extension::Degree2ExtensionField, sqrt},
        point::ShortWeierstrassJacobianPoint,
        traits::Compress,
    },
    field::element::FieldElement,
};
use core::cmp::Ordering;

use crate::{
    cyclic_group::IsGroup, elliptic_curve::traits::FromAffine, errors::ByteConversionError,
    traits::ByteConversion,
};

type G1Point = ShortWeierstrassJacobianPoint<BLS12381Curve>;
type G2Point = ShortWeierstrassJacobianPoint<BLS12381TwistCurve>;
type BLS12381FieldElement = FieldElement<BLS12381PrimeField>;

/// This functionality includes the compression and decompression for points belonging to the BLS12-381, following the ideas in
/// Zcash curve compression. See <https://hackmd.io/@benjaminion/bls12-381#Point-compression> and
/// <https://github.com/zcash/librustzcash/blob/6e0364cd42a2b3d2b958a54771ef51a8db79dd29/pairing/src/bls12_381/README.md#serialization>.
/// The way we encode points differs from the one used ordinarily for serialization in lambdaworks.
/// G1 points are represented by their x coordinate in big-endian form (48 bytes), with the three most significant bits used to give information on the compressed format, whether the point is the point at infinity and which of the two roots to take
/// G2 points are represented by their x coordinate in big-endian form (96 bytes), following the order a * i + b. The three most significant bits contain the same type of information as in G1.
impl Compress for BLS12381Curve {
    type G1Point = G1Point;

    type G2Point = G2Point;

    type G1Compressed = [u8; 48];

    type G2Compressed = [u8; 96];

    type Error = ByteConversionError;

    #[cfg(feature = "alloc")]
    fn compress_g1_point(point: &Self::G1Point) -> Self::G1Compressed {
        if *point == G1Point::neutral_element() {
            // point is at infinity
            let mut x_bytes = [0_u8; 48];
            x_bytes[0] |= 1 << 7;
            x_bytes[0] |= 1 << 6;
            x_bytes
        } else {
            // point is not at infinity
            let point_affine = point.to_affine();
            let x = point_affine.x();
            let y = point_affine.y();
            let mut x_bytes = [0u8; 48];
            let bytes = x.to_bytes_be();
            x_bytes.copy_from_slice(&bytes);

            // Set first bit to to 1 indicate this is compressed element.
            x_bytes[0] |= 1 << 7;

            let y_neg = core::ops::Neg::neg(y);
            if y_neg.canonical() < y.canonical() {
                x_bytes[0] |= 1 << 5;
            }
            x_bytes
        }
    }

    fn decompress_g1_point(input_bytes: &mut [u8]) -> Result<Self::G1Point, Self::Error> {
        if input_bytes.len() != 48 {
            return Err(ByteConversionError::InvalidValue);
        }
        let first_byte = input_bytes.first().unwrap();
        // We get the 3 most significant bits
        let prefix_bits = first_byte >> 5;
        let first_bit = (prefix_bits & 4_u8) >> 2;
        // If first bit is not 1, then the value is not compressed.
        if first_bit != 1 {
            return Err(ByteConversionError::ValueNotCompressed);
        }
        let second_bit = (prefix_bits & 2_u8) >> 1;
        // If the second bit is 1, then the compressed point is the
        // point at infinity. Validate canonical encoding: remaining bits/bytes must be zero.
        if second_bit == 1 {
            if (first_byte & 0x1f) != 0 || input_bytes[1..].iter().any(|&b| b != 0) {
                return Err(ByteConversionError::InvalidValue);
            }
            return Ok(G1Point::neutral_element());
        }
        // We obtain the third bit
        let third_bit = prefix_bits & 1_u8;

        let first_byte_without_control_bits = (first_byte << 3) >> 3;
        input_bytes[0] = first_byte_without_control_bits;

        let x = BLS12381FieldElement::from_bytes_be(input_bytes)?;

        // We apply the elliptic curve formula to know the y^2 value.
        let y_squared = x.pow(3_u16) + BLS12381FieldElement::from(4);

        let (y_sqrt_1, y_sqrt_2) = &y_squared.sqrt().ok_or(ByteConversionError::InvalidValue)?;

        // we call "negative" to the greate root,
        // if the third bit is 1, we take this grater value.
        // Otherwise, we take the second one.
        let y = match (y_sqrt_1.canonical().cmp(&y_sqrt_2.canonical()), third_bit) {
            (Ordering::Greater, 0) => y_sqrt_2,
            (Ordering::Greater, _) => y_sqrt_1,
            (Ordering::Less, 0) => y_sqrt_1,
            (Ordering::Less, _) => y_sqrt_2,
            (Ordering::Equal, _) => y_sqrt_1,
        };

        let point =
            G1Point::from_affine(x, y.clone()).map_err(|_| ByteConversionError::InvalidValue)?;

        point
            .is_in_subgroup()
            .then_some(point)
            .ok_or(ByteConversionError::PointNotInSubgroup)
    }

    #[cfg(feature = "alloc")]
    fn compress_g2_point(point: &Self::G2Point) -> Self::G2Compressed {
        if *point == G2Point::neutral_element() {
            // point is at infinity
            let mut x_bytes = [0_u8; 96];
            x_bytes[0] |= 1 << 7;
            x_bytes[0] |= 1 << 6;
            x_bytes
        } else {
            // point is not at infinity
            let point_affine = point.to_affine();
            let x = point_affine.x();
            let y = point_affine.y();

            let x_rev: FieldElement<Degree2ExtensionField> =
                FieldElement::new([x.value()[1].clone(), x.value()[0].clone()]);
            let mut x_bytes = [0u8; 96];
            let bytes = x_rev.to_bytes_be();
            x_bytes.copy_from_slice(&bytes);

            // Set first bit to to 1 indicate this is compressed element.
            x_bytes[0] |= 1 << 7;

            // Set the 3rd bit based on y value.
            let y_neg = -y;

            match (
                y.value()[0].canonical().cmp(&y_neg.value()[0].canonical()),
                y.value()[1].canonical().cmp(&y_neg.value()[1].canonical()),
            ) {
                (Ordering::Greater, _) | (Ordering::Equal, Ordering::Greater) => {
                    x_bytes[0] |= 1 << 5;
                }
                (_, _) => (),
            }
            x_bytes
        }
    }

    fn decompress_g2_point(input_bytes: &mut [u8]) -> Result<Self::G2Point, Self::Error> {
        if input_bytes.len() != 96 {
            return Err(ByteConversionError::InvalidValue);
        }

        let first_byte = input_bytes.first().unwrap();

        // We get the first 3 bits
        let prefix_bits = first_byte >> 5;
        let first_bit = (prefix_bits & 4_u8) >> 2;
        // If first bit is not 1, then the value is not compressed.
        if first_bit != 1 {
            return Err(ByteConversionError::InvalidValue);
        }
        let second_bit = (prefix_bits & 2_u8) >> 1;
        // If the second bit is 1, then the compressed point is the
        // point at infinity. Validate canonical encoding: remaining bits/bytes must be zero.
        if second_bit == 1 {
            if (first_byte & 0x1f) != 0 || input_bytes[1..].iter().any(|&b| b != 0) {
                return Err(ByteConversionError::InvalidValue);
            }
            return Ok(Self::G2Point::neutral_element());
        }

        let third_bit = prefix_bits & 1_u8;

        let first_byte_without_control_bits = (first_byte << 3) >> 3;
        input_bytes[0] = first_byte_without_control_bits;

        let input0 = &input_bytes[48..];
        let input1 = &input_bytes[0..48];
        let x0 = BLS12381FieldElement::from_bytes_be(input0)?;
        let x1 = BLS12381FieldElement::from_bytes_be(input1)?;
        let x: FieldElement<Degree2ExtensionField> = FieldElement::new([x0, x1]);

        const VALUE: BLS12381FieldElement = BLS12381FieldElement::from_hex_unchecked("4");
        let b_param_qfe = FieldElement::<Degree2ExtensionField>::new([VALUE, VALUE]);

        let y = sqrt::sqrt_qfe(&(x.pow(3_u64) + b_param_qfe), third_bit)
            .ok_or(ByteConversionError::InvalidValue)?;

        let point =
            Self::G2Point::from_affine(x, y).map_err(|_| ByteConversionError::InvalidValue)?;

        point
            .is_in_subgroup()
            .then_some(point)
            .ok_or(ByteConversionError::PointNotInSubgroup)
    }
}

#[cfg(test)]
mod tests {
    use super::{BLS12381FieldElement, G1Point};
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
    use crate::elliptic_curve::short_weierstrass::traits::Compress;
    use crate::elliptic_curve::traits::{FromAffine, IsEllipticCurve};

    #[cfg(feature = "alloc")]
    use crate::{
        cyclic_group::IsGroup, traits::ByteConversion, unsigned_integer::element::UnsignedInteger,
    };

    #[test]
    fn test_zero_point() {
        let g1 = BLS12381Curve::generator();

        assert!(g1.is_in_subgroup());
        let new_x = BLS12381FieldElement::zero();
        let new_y = BLS12381FieldElement::one() + BLS12381FieldElement::one();

        let false_point2 = G1Point::from_affine(new_x, new_y).unwrap();

        assert!(!false_point2.is_in_subgroup());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g1_compress_generator() {
        use crate::elliptic_curve::short_weierstrass::traits::Compress;

        let g = BLS12381Curve::generator();
        let mut compressed_g = BLS12381Curve::compress_g1_point(&g);
        let first_byte = compressed_g.first().unwrap();

        let first_byte_without_control_bits = (first_byte << 3) >> 3;
        compressed_g[0] = first_byte_without_control_bits;

        let compressed_g_x = BLS12381FieldElement::from_bytes_be(&compressed_g).unwrap();
        let g_x = g.x();

        assert_eq!(*g_x, compressed_g_x);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_g1_compress_point_at_inf() {
        use crate::elliptic_curve::short_weierstrass::traits::Compress;

        let inf = G1Point::neutral_element();
        let compressed_inf = BLS12381Curve::compress_g1_point(&inf);
        let first_byte = compressed_inf.first().unwrap();

        assert_eq!(*first_byte >> 6, 3_u8);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_generator() {
        use crate::elliptic_curve::short_weierstrass::traits::Compress;

        let g = BLS12381Curve::generator();
        let mut compressed_g_slice = BLS12381Curve::compress_g1_point(&g);

        let decompressed_g = BLS12381Curve::decompress_g1_point(&mut compressed_g_slice).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_2g() {
        let g = BLS12381Curve::generator();
        // calculate g point operate with itself
        let g_2 = g.operate_with_self(UnsignedInteger::<4>::from("2"));

        let mut compressed_g2_slice: [u8; 48] = BLS12381Curve::compress_g1_point(&g_2);

        let decompressed_g2 = BLS12381Curve::decompress_g1_point(&mut compressed_g2_slice).unwrap();

        assert_eq!(g_2, decompressed_g2);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_generator_g2() {
        use crate::elliptic_curve::short_weierstrass::{
            curves::bls12_381::twist::BLS12381TwistCurve, traits::Compress,
        };

        let g = BLS12381TwistCurve::generator();
        let mut compressed_g_slice = BLS12381Curve::compress_g2_point(&g);

        let decompressed_g = BLS12381Curve::decompress_g2_point(&mut compressed_g_slice).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_generator_g2_neg() {
        use crate::elliptic_curve::short_weierstrass::{
            curves::bls12_381::twist::BLS12381TwistCurve, traits::Compress,
        };

        let g = BLS12381TwistCurve::generator();
        let g_neg = g.neg();

        let mut compressed_g_neg_slice = BLS12381Curve::compress_g2_point(&g_neg);

        let decompressed_g_neg =
            BLS12381Curve::decompress_g2_point(&mut compressed_g_neg_slice).unwrap();

        assert_eq!(g_neg, decompressed_g_neg);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_2g_g2() {
        use crate::elliptic_curve::short_weierstrass::{
            curves::bls12_381::twist::BLS12381TwistCurve, traits::Compress,
        };

        let g = BLS12381TwistCurve::generator();
        // Use 2*g to test with a non-generator subgroup point
        let g_2 = g.operate_with_self(UnsignedInteger::<4>::from("2"));

        let mut compressed_g2_slice: [u8; 96] = BLS12381Curve::compress_g2_point(&g_2);

        let decompressed_g2 = BLS12381Curve::decompress_g2_point(&mut compressed_g2_slice).unwrap();

        assert_eq!(g_2, decompressed_g2);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_compress_decompress_3g_g2() {
        use crate::elliptic_curve::short_weierstrass::{
            curves::bls12_381::twist::BLS12381TwistCurve, traits::Compress,
        };

        let g = BLS12381TwistCurve::generator();
        // Use 3*g to test with another subgroup point
        let g_3 = g.operate_with_self(UnsignedInteger::<4>::from("3"));

        let mut compressed_g3_slice: [u8; 96] = BLS12381Curve::compress_g2_point(&g_3);

        let decompressed_g3 = BLS12381Curve::decompress_g2_point(&mut compressed_g3_slice).unwrap();

        assert_eq!(g_3, decompressed_g3);
    }
}
