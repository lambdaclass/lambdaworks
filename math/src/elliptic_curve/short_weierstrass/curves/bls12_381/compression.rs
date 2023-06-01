use super::curve::{BLS12381Curve, BLS12381FieldElement};
use super::twist::BLS12381TwistCurve;
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::{
    short_weierstrass::curves::bls12_381::{
        default_types::FrConfig, field_extension::Degree2ExtensionField,
    },
    short_weierstrass::point::ShortWeierstrassProjectivePoint,
    traits::FromAffine,
    traits::{Compress, EllipticCurveError, IsEllipticCurve},
};
use crate::field::element::FieldElement;
use crate::field::fields::montgomery_backed_prime_fields::IsModulus;
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::UnsignedInteger;
use std::ops::Neg;

fn check_point_is_in_subgroup(point: &ShortWeierstrassProjectivePoint<BLS12381Curve>) -> bool {
    const MODULUS: UnsignedInteger<4> = FrConfig::MODULUS;
    let inf = ShortWeierstrassProjectivePoint::<BLS12381Curve>::neutral_element();
    let aux_point = point.operate_with_self(MODULUS);
    inf == aux_point
}

impl Compress for ShortWeierstrassProjectivePoint<BLS12381Curve> {
    type G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
    type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

    fn compress_g1_point(point: &Self::G1Point) -> Result<[u8; 48], EllipticCurveError> {
        let ret_vec = if *point == Self::G1Point::neutral_element() {
            // point is at infinity
            let mut x_bytes = vec![0_u8; 48];
            x_bytes[0] |= 1 << 7;
            x_bytes[0] |= 1 << 6;
            x_bytes
        } else {
            // point is not at infinity
            let point_affine = point.to_affine();
            let x = point_affine.x();
            let y = point_affine.y();

            let mut x_bytes = x.to_bytes_be();

            // Set first bit to to 1 indicate this is compressed element.
            x_bytes[0] |= 1 << 7;

            let y_neg = y.neg();
            if y_neg.representative() < y.representative() {
                x_bytes[0] |= 1 << 5;
            }
            x_bytes
        };
        ret_vec
            .try_into()
            .map_err(|_e| EllipticCurveError::InvalidPoint)
    }

    fn decompress_g1_point(
        input_bytes: &mut [u8; 48],
    ) -> Result<Self::G1Point, EllipticCurveError> {
        let first_byte = input_bytes.first().unwrap();
        // We get the first 3 bits
        let prefix_bits = first_byte >> 5;
        let first_bit = (prefix_bits & 4_u8) >> 2;
        // If first bit is not 1, then the value is not compressed.
        if first_bit != 1 {
            return Err(EllipticCurveError::InvalidPoint);
        }
        let second_bit = (prefix_bits & 2_u8) >> 1;
        // If the second bit is 1, then the compressed point is the
        // point at infinity and we return it directly.
        if second_bit == 1 {
            return Ok(Self::G1Point::neutral_element());
        }
        let third_bit = prefix_bits & 1_u8;

        let first_byte_without_contorl_bits = (first_byte << 3) >> 3;
        input_bytes[0] = first_byte_without_contorl_bits;

        let x = BLS12381FieldElement::from_bytes_be(input_bytes)
            .map_err(|_e| EllipticCurveError::InvalidPoint)?;

        // We apply the elliptic curve formula to know the y^2 value.
        let y_squared = x.pow(3_u16) + BLS12381FieldElement::from(4);

        let (y_sqrt_1, y_sqrt_2) = &y_squared.sqrt().ok_or(EllipticCurveError::InvalidPoint)?;

        // we call "negative" to the greate root,
        // if the third bit is 1, we take this grater value.
        // Otherwise, we take the second one.
        let y = super::sqrt::select_sqrt_value_from_third_bit(
            y_sqrt_1.clone(),
            y_sqrt_2.clone(),
            third_bit,
        );
        let point =
            Self::G1Point::from_affine(x, y).map_err(|_| EllipticCurveError::InvalidPoint)?;

        check_point_is_in_subgroup(&point)
            .then_some(point)
            .ok_or(EllipticCurveError::InvalidPoint)
    }

    fn decompress_g2_point(
        input_bytes: &mut [u8; 96],
    ) -> Result<Self::G2Point, EllipticCurveError> {
        let first_byte = input_bytes.first().unwrap();

        // We get the first 3 bits
        let prefix_bits = first_byte >> 5;
        let first_bit = (prefix_bits & 4_u8) >> 2;
        // If first bit is not 1, then the value is not compressed.
        if first_bit != 1 {
            return Err(EllipticCurveError::InvalidPoint);
        }
        let second_bit = (prefix_bits & 2_u8) >> 1;
        // If the second bit is 1, then the compressed point is the
        // point at infinity and we return it directly.
        if second_bit == 1 {
            return Ok(Self::G2Point::neutral_element());
        }

        let first_byte_without_contorl_bits = (first_byte << 3) >> 3;
        input_bytes[0] = first_byte_without_contorl_bits;

        let input0 = &input_bytes[48..];
        let input1 = &input_bytes[0..48];
        let x0 = BLS12381FieldElement::from_bytes_be(input0).unwrap();
        let x1 = BLS12381FieldElement::from_bytes_be(input1).unwrap();
        let x: FieldElement<Degree2ExtensionField> = FieldElement::new([x0, x1]);

        const VALUE: BLS12381FieldElement = BLS12381FieldElement::from_hex_unchecked("4");
        let b_param_qfe = FieldElement::<Degree2ExtensionField>::new([VALUE, VALUE]);

        let y = super::sqrt::sqrt_qfe(&(x.pow(3_u64) + b_param_qfe), 0)
            .ok_or(EllipticCurveError::InvalidPoint)?;
        Self::G2Point::from_affine(x, y).map_err(|_| EllipticCurveError::InvalidPoint)
    }
}

#[cfg(test)]
mod tests {
    use super::super::curve::{BLS12381Curve, BLS12381FieldElement};
    use super::Compress;
    use crate::cyclic_group::IsGroup;
    use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
    use crate::elliptic_curve::traits::{FromAffine, IsEllipticCurve};
    use crate::traits::ByteConversion;
    use crate::unsigned_integer::element::UnsignedInteger;

    type G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;

    #[test]
    fn test_zero_point() {
        let g1 = BLS12381Curve::generator();

        assert!(super::check_point_is_in_subgroup(&g1));
        let new_x = BLS12381FieldElement::zero();
        let new_y = BLS12381FieldElement::one() + BLS12381FieldElement::one();

        let false_point2 = G1Point::from_affine(new_x, new_y).unwrap();

        assert!(!super::check_point_is_in_subgroup(&false_point2));
    }

    #[test]
    fn test_g1_compress_generator() {
        let g = BLS12381Curve::generator();
        let mut compressed_g = G1Point::compress_g1_point(&g).unwrap();
        let first_byte = compressed_g[0];

        let first_byte_without_control_bits = (first_byte << 3) >> 3;
        compressed_g[0] = first_byte_without_control_bits;

        let compressed_g_x = BLS12381FieldElement::from_bytes_be(&compressed_g).unwrap();
        let g_x = g.x();

        assert_eq!(*g_x, compressed_g_x);
    }

    #[test]
    fn test_g1_compress_point_at_inf() {
        let inf = G1Point::neutral_element();
        let compressed_inf = G1Point::compress_g1_point(&inf).unwrap();
        let first_byte = compressed_inf[0];

        assert_eq!(first_byte >> 6, 3_u8);
    }

    #[test]
    fn test_compress_decompress_generator() {
        let g = BLS12381Curve::generator();
        let mut compressed_g = G1Point::compress_g1_point(&g).unwrap();
        let decompressed_g = G1Point::decompress_g1_point(&mut compressed_g).unwrap();

        assert_eq!(g, decompressed_g);
    }

    #[test]
    fn test_compress_decompress_2g() {
        let g = BLS12381Curve::generator();
        // calculate g point operate with itself
        let g_2 = g.operate_with_self(UnsignedInteger::<4>::from("2"));
        let mut compressed_g2 = G1Point::compress_g1_point(&g_2).unwrap();
        let decompressed_g2 = G1Point::decompress_g1_point(&mut compressed_g2).unwrap();

        assert_eq!(g_2, decompressed_g2);
    }

    #[test]
    fn short_test_compress_and_decompress_point() {
        let line = "8d0c6eeadd3f8529d67246f77404a4ac2d9d7fd7d50cf103d3e6abb9003e5e36d8f322663ebced6707a7f46d97b7566d";
        let bytes = hex::decode(line).unwrap();
        let mut input_bytes: [u8; 48] = bytes.try_into().unwrap();
        let point = G1Point::decompress_g1_point(&mut input_bytes).unwrap();
        let compressed = G1Point::compress_g1_point(&point).unwrap();
        let hex_string = hex::encode(compressed);

        assert_eq!("8d0c6eeadd3f8529d67246f77404a4ac2d9d7fd7d50cf103d3e6abb9003e5e36d8f322663ebced6707a7f46d97b7566d", &hex_string);
    }
}
