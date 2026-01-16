use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::fields::secp256r1_field::Secp256r1PrimeField;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

/// This implementation is not constant time and cannot be used to sign messages. You can use it to check signatures
#[derive(Clone, Debug)]
pub struct Secp256r1Curve;

impl IsEllipticCurve for Secp256r1Curve {
    type BaseField = Secp256r1PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator point is mathematically verified to be a valid point on the curve.
        // - `unwrap()` is safe because the provided coordinates satisfy the curve equation.
        let point = Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296",
            ),
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5",
            ),
            FieldElement::one(),
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for Secp256r1Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::from_hex_unchecked(
            "ffffffff00000001000000000000000000000000fffffffffffffffffffffffc",
        )
    }
    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::from_hex_unchecked(
            "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U256,
    };

    use super::Secp256r1Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FE = FieldElement<Secp256r1PrimeField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<Secp256r1Curve> {
        let x = FE::from_hex_unchecked(
            "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296",
        );
        let y = FE::from_hex_unchecked(
            "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5",
        );
        Secp256r1Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<Secp256r1Curve> {
        let x = FE::from_hex_unchecked(
            "51590B7A515140D2D784C85608668FDFEF8C82FD1F5BE52421554A0DC3D033ED",
        );
        let y = FE::from_hex_unchecked(
            "E0C17DA8904A727D8AE1BF36BF8A79260D012F00D4D80888D1D0BB44FDA16DA4",
        );
        Secp256r1Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_1_works() {
        let point_1 = point_1();
        let point_1_times_5 = point_1_times_5();
        assert_eq!(point_1.operate_with_self(5_u16), point_1_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(
            *p.x(),
            FE::from_hex_unchecked(
                "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"
            )
        );
        assert_eq!(
            *p.y(),
            FE::from_hex_unchecked(
                "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"
            )
        );
        assert_eq!(*p.z(), FE::from_hex_unchecked("1"));
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            Secp256r1Curve::create_point_from_affine(FE::from(0), FE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = Secp256r1Curve::generator();
        let g2 = g.operate_with_self(2_u16);
        let g2_other = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
        assert_eq!(&g2, &g2_other);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = Secp256r1Curve::generator();
        let g2 = g.operate_with_self(2_u16);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of Secp256r1 curve equation
        let a = Secp256r1Curve::a();
        let b = Secp256r1Curve::b();
        let y_sq_0 = x.pow(3_u16) + (a * x) + b;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = Secp256r1Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn generator_has_right_order() {
        let g = Secp256r1Curve::generator();
        assert_eq!(
            g.operate_with_self(U256::from_hex_unchecked(
                "0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"
            ))
            .to_affine(),
            ShortWeierstrassProjectivePoint::neutral_element()
        );
    }

    #[test]
    fn inverse_works() {
        let g = Secp256r1Curve::generator();
        assert_eq!(
            g.operate_with_self(U256::from_hex_unchecked(
                "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC63254C"
            ))
            .to_affine(),
            g.operate_with_self(5u64).neg().to_affine()
        );
    }

    #[test]
    fn addition_with_neutral_element_returns_same_element() {
        let g = Secp256r1Curve::generator();
        let p = g.operate_with_self(12345u64);
        let neutral = ShortWeierstrassProjectivePoint::<Secp256r1Curve>::neutral_element();

        assert_eq!(p.operate_with(&neutral), p);
        assert_eq!(neutral.operate_with(&p), p);
    }

    #[test]
    fn neutral_element_plus_neutral_element_is_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<Secp256r1Curve>::neutral_element();
        assert_eq!(neutral.operate_with(&neutral), neutral);
    }

    #[test]
    fn add_opposite_of_a_point_to_itself_gives_neutral_element() {
        let g = Secp256r1Curve::generator();
        let p = g.operate_with_self(54321u64);
        let neg_p = p.neg();
        let result = p.operate_with(&neg_p);
        let neutral = ShortWeierstrassProjectivePoint::<Secp256r1Curve>::neutral_element();
        assert_eq!(result, neutral);
    }

    #[test]
    fn doubling_neutral_element_gives_neutral_element() {
        let neutral = ShortWeierstrassProjectivePoint::<Secp256r1Curve>::neutral_element();
        assert_eq!(neutral.operate_with_self(2u64), neutral);
    }

    #[test]
    fn scalar_mul_by_zero_gives_neutral_element() {
        let g = Secp256r1Curve::generator();
        let result = g.operate_with_self(0u64);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn scalar_mul_by_one_gives_same_point() {
        let g = Secp256r1Curve::generator();
        let result = g.operate_with_self(1u64);
        assert_eq!(result.to_affine(), g.to_affine());
    }

    #[test]
    fn associativity_holds() {
        let g = Secp256r1Curve::generator();
        let a = g.operate_with_self(111u64);
        let b = g.operate_with_self(222u64);
        let c = g.operate_with_self(333u64);

        let left = a.operate_with(&b).operate_with(&c);
        let right = a.operate_with(&b.operate_with(&c));
        assert_eq!(left.to_affine(), right.to_affine());
    }

    #[test]
    fn commutativity_holds() {
        let g = Secp256r1Curve::generator();
        let a = g.operate_with_self(12345u64);
        let b = g.operate_with_self(67890u64);

        assert_eq!(
            a.operate_with(&b).to_affine(),
            b.operate_with(&a).to_affine()
        );
    }

    #[test]
    fn distributivity_of_scalar_mul() {
        let g = Secp256r1Curve::generator();
        let a = 100u64;
        let b = 200u64;

        // g * (a + b) = g * a + g * b
        let left = g.operate_with_self(a + b);
        let right = g.operate_with_self(a).operate_with(&g.operate_with_self(b));
        assert_eq!(left.to_affine(), right.to_affine());
    }
}
