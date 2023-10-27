use super::field_extension::BN254PrimeField;

use crate::elliptic_curve::short_weierstrass::curves::bls12_377::curve::BLS12377Curve;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;

use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

// Curve description

#[derive(Clone, Debug)]

pub struct BN254Curve;

impl IsEllipticCurve for BN254Curve {
    type BaseField = BN254PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    //Affine Coordinates
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            // I don't believe in this ut i gotta represent this in the other file
            FieldElement::<Self::BaseField>::new_base("1"),
            FieldElement::<Self::BaseField>::new_base("2"),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for BN254Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(3)
    }
}

#[cfg(test)]
mod test {
    use crate::cyclic_group::IsGroup;
    use crate::elliptic_curve::{
        short_weierstrass::curves::bn254::curve::BN254Curve, traits::IsEllipticCurve,
    };

    #[test]
    fn operate_with_self_works_1() {
        let g = BN254Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }
}
