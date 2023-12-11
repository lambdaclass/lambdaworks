use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::fields::vesta_field::Vesta255PrimeField;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

#[derive(Clone, Debug)]
pub struct VestaCurve;

impl IsEllipticCurve for VestaCurve {
    type BaseField = Vesta255PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            -FieldElement::<Self::BaseField>::one(),
            FieldElement::<Self::BaseField>::from(2),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for VestaCurve {
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

    use super::VestaCurve;

    #[allow(clippy::upper_case_acronyms)]
    type FE = FieldElement<Vesta255PrimeField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<VestaCurve> {
        let x = FE::from_hex_unchecked(
            "c4e6a8789457a64e1638783181963d4c4399a5a8cdb30af4038664ce431033c",
        );
        let y = FE::from_hex_unchecked(
            "2d8c9125be9a3ac50371e462f63dfc3fbbf645e9a93d6b7da71c13d3065e3ce5",
        );
        VestaCurve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<VestaCurve> {
        let x = FE::from_hex_unchecked(
            "1266f29f1478410eaa62fb1ab064f7d9259f515600544165972a89c9941c72c3",
        );
        let y = FE::from_hex_unchecked(
            "3a893b592bd487cd25c5d4237b02987e1b78206e70989f3209e24a40b89499fd",
        );
        VestaCurve::create_point_from_affine(x, y).unwrap()
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
                "c4e6a8789457a64e1638783181963d4c4399a5a8cdb30af4038664ce431033c"
            )
        );
        assert_eq!(
            *p.y(),
            FE::from_hex_unchecked(
                "2d8c9125be9a3ac50371e462f63dfc3fbbf645e9a93d6b7da71c13d3065e3ce5"
            )
        );
        assert_eq!(*p.z(), FE::from_hex_unchecked("1"));
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            VestaCurve::create_point_from_affine(FE::from(0), FE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = VestaCurve::generator();
        let g2 = g.operate_with_self(2_u16);
        let g2_other = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
        assert_eq!(&g2, &g2_other);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = VestaCurve::generator();
        let g2 = g.operate_with_self(2_u16);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of Pallas curve equation
        let five = VestaCurve::b();
        let y_sq_0 = x.pow(3_u16) + five;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = VestaCurve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }
}
