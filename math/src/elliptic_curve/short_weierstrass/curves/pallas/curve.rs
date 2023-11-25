use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::fields::pallas_field::Pallas255PrimeField;
use crate::unsigned_integer::element::UnsignedInteger;
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
            FieldElement::<Self::BaseField>::from_raw(&UnsignedInteger {
                limbs: [2, 0, 0, 0],
            }),
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
        let x = FE::from_hex_unchecked("bd1e740e6b1615ae4c508148ca0c53dbd43f7b2e206195ab638d7f45d51d6b5");
        let y = FE::from_hex_unchecked("13aacd107ca10b7f8aab570da1183b91d7d86dd723eaa2306b0ef9c5355b91d8");
        PallasCurve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<PallasCurve> {
        let x = FE::from_hex_unchecked("17a21304fffd6749d6173d4e0acd9724d98a97453b3491c0e5a53b06cf039b13");
        let y = FE::from_hex_unchecked("2f9bde429091a1089e52a6cc5dc789e1a58eeded0cf72dccc33b7af685a982d");
        PallasCurve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_1_works() {
        let point_1 = point_1();
        let point_1_times_5 = point_1_times_5();
        assert_eq!(point_1.operate_with_self(5_u16), point_1_times_5);
    }
}
