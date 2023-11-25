use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::fields::pallas_field::Pallas255PrimeField;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

#[derive(Clone, Debug)]
pub struct PallasCurve;

impl IsEllipticCurve for PallasCurve {
    type BaseField = Pallas255PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            -FieldElement::<Self::BaseField>::one(),
            FieldElement::<Self::BaseField>::from(2),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for PallasCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement,
    };

    use super::PallasCurve;

    #[allow(clippy::upper_case_acronyms)]
    type FE = FieldElement<Pallas255PrimeField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<PallasCurve> {
        let x = FE::from_hex_unchecked(
            "bd1e740e6b1615ae4c508148ca0c53dbd43f7b2e206195ab638d7f45d51d6b5",
        );
        let y = FE::from_hex_unchecked(
            "13aacd107ca10b7f8aab570da1183b91d7d86dd723eaa2306b0ef9c5355b91d8",
        );
        PallasCurve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<PallasCurve> {
        let x = FE::from_hex_unchecked(
            "17a21304fffd6749d6173d4e0acd9724d98a97453b3491c0e5a53b06cf039b13",
        );
        let y = FE::from_hex_unchecked(
            "2f9bde429091a1089e52a6cc5dc789e1a58eeded0cf72dccc33b7af685a982d",
        );
        PallasCurve::create_point_from_affine(x, y).unwrap()
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
                "bd1e740e6b1615ae4c508148ca0c53dbd43f7b2e206195ab638d7f45d51d6b5"
            )
        );
        assert_eq!(
            *p.y(),
            FE::from_hex_unchecked(
                "13aacd107ca10b7f8aab570da1183b91d7d86dd723eaa2306b0ef9c5355b91d8"
            )
        );
        assert_eq!(*p.z(), FE::from_hex_unchecked("1"));
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            PallasCurve::create_point_from_affine(FE::from(0), FE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = PallasCurve::generator();
        let g2 = g.operate_with_self(2_u16);
        let g2_other = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
        assert_eq!(&g2, &g2_other);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = PallasCurve::generator();
        let g2 = g.operate_with_self(2_u16);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of Pallas curve equation
        let five = PallasCurve::b();
        let y_sq_0 = x.pow(3_u16) + five;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = PallasCurve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }
}
