use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::fields::secp256k1_scalarfield::Secp256k1ScalarField;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

#[derive(Clone, Debug)]
pub struct Secq256k1Curve;

impl IsEllipticCurve for Secq256k1Curve {
    type BaseField = Secp256k1ScalarField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator point is mathematically verified to be a valid point on the curve.
        // - `unwrap()` is safe because the provided coordinates satisfy the curve equation.
        let point = Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "76C39F5585CB160EB6B06C87A2CE32E23134E45A097781A6A24288E37702EDA6",
            ),
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "3FFC646C7B2918B5DC2D265A8E82A7F7D18983D26E8DC055A4120DDAD952677F",
            ),
            FieldElement::one(),
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for Secq256k1Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U256,
    };

    use super::Secq256k1Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FE = FieldElement<Secp256k1ScalarField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<Secq256k1Curve> {
        let x = FE::from_hex_unchecked(
            "76C39F5585CB160EB6B06C87A2CE32E23134E45A097781A6A24288E37702EDA6",
        );
        let y = FE::from_hex_unchecked(
            "3FFC646C7B2918B5DC2D265A8E82A7F7D18983D26E8DC055A4120DDAD952677F",
        );
        Secq256k1Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<Secq256k1Curve> {
        let x = FE::from_hex_unchecked(
            "8656a2c13dd0a3bfa362d2ff8c00281341ff3a79cbbe8857f2d20b398041a21a",
        );
        let y = FE::from_hex_unchecked(
            "468ed8bcfcd4ed2b3bf154414b9e48d8c5ce54f6616846a7cf6a725f70d34a63",
        );
        let z = FE::from_hex_unchecked(
            "bb26eae3d2b9603d98dff86d87175f442e539c07bbe4ef5712e47c4d72c89734",
        );
        ShortWeierstrassProjectivePoint::<Secq256k1Curve>::new([x, y, z]).unwrap()
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
                "76C39F5585CB160EB6B06C87A2CE32E23134E45A097781A6A24288E37702EDA6"
            )
        );
        assert_eq!(
            *p.y(),
            FE::from_hex_unchecked(
                "3FFC646C7B2918B5DC2D265A8E82A7F7D18983D26E8DC055A4120DDAD952677F"
            )
        );
        assert_eq!(*p.z(), FE::from_hex_unchecked("1"));
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            Secq256k1Curve::create_point_from_affine(FE::from(0), FE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = Secq256k1Curve::generator();
        let g2 = g.operate_with_self(2_u16);
        let g2_other = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
        assert_eq!(&g2, &g2_other);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = Secq256k1Curve::generator();
        let g2 = g.operate_with_self(2_u16);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of secq256k1 curve equation
        let seven = Secq256k1Curve::b();
        let y_sq_0 = x.pow(3_u16) + seven;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works() {
        let g = Secq256k1Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn generator_has_right_order() {
        let g = Secq256k1Curve::generator();
        assert_eq!(
            g.operate_with_self(U256::from_hex_unchecked(
                "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
            ))
            .to_affine(),
            ShortWeierstrassProjectivePoint::neutral_element()
        );
    }

    #[test]
    /// (r - 5)g = rg - 5g = 0 - 5g = -5g
    fn inverse_works() {
        let g = Secq256k1Curve::generator();
        assert_eq!(
            g.operate_with_self(U256::from_hex_unchecked(
                "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2A"
            ))
            .to_affine(),
            g.operate_with_self(5u64).neg().to_affine()
        );
    }
}
