use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::fields::secq256k1_field::Secq256k1PrimeField;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

#[derive(Clone, Debug)]
pub struct Secq256k1Curve;

impl IsEllipticCurve for Secq256k1Curve {
    type BaseField = Secq256k1PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "76C39F5585CB160EB6B06C87A2CE32E23134E45A097781A6A24288E37702EDA6",
            ),
            FieldElement::<Self::BaseField>::from_hex_unchecked(
                "3FFC646C7B2918B5DC2D265A8E82A7F7D18983D26E8DC055A4120DDAD952677F",
            ),
            FieldElement::one(),
        ])
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
    type FE = FieldElement<Secq256k1PrimeField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<Secq256k1Curve> {
        let x = FE::from_hex_unchecked(
            "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
        );
        let y = FE::from_hex_unchecked(
            "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
        );
        Secq256k1Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<Secq256k1Curve> {
        let x = FE::from_hex_unchecked(
            "2F8BDE4D1A07209355B4A7250A5C5128E88B84BDDC619AB7CBA8D569B240EFE4",
        );
        let y = FE::from_hex_unchecked(
            "D8AC222636E5E3D6D4DBA9DDA6C9C426F788271BAB0D6840DCA87D3AA6AC62D6",
        );
        Secq256k1Curve::create_point_from_affine(x, y).unwrap()
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
                "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
            )
        );
        assert_eq!(
            *p.y(),
            FE::from_hex_unchecked(
                "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
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

        // calculate both sides of secp256k1 curve equation
        let seven = Secq256k1Curve::b();
        let y_sq_0 = x.pow(3_u16) + seven;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
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
                "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"
            ))
            .to_affine(),
            ShortWeierstrassProjectivePoint::neutral_element()
        );
    }

    #[test]
    fn inverse_works() {
        let g = Secq256k1Curve::generator();
        assert_eq!(
            g.operate_with_self(U256::from_hex_unchecked(
                "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd036413C"
            ))
            .to_affine(),
            g.operate_with_self(5u64).neg().to_affine()
        );
    }
}
