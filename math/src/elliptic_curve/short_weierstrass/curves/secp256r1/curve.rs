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
        unsafe {
            Self::PointRepresentation::new([
                FieldElement::<Self::BaseField>::from_hex_unchecked(
                    "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296",
                ),
                FieldElement::<Self::BaseField>::from_hex_unchecked(
                    "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5",
                ),
                FieldElement::one(),
            ])
            .unwrap_unchecked()
        }
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
}
