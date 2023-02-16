use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use crate::field::fields::u64_prime_field::U64PrimeField;
use std::fmt::Debug;

use super::element::EdwardsProjectivePoint;

/// Trait to add elliptic curves behaviour to a struct.
pub trait IsEdwards: IsEllipticCurve + Clone + Debug {
    fn a() -> FieldElement<Self::BaseField>;

    fn d() -> FieldElement<Self::BaseField>;

    /// Evaluates the short Weierstrass equation at (x, y z).
    /// Used for checking if [x: y: z] belongs to the elliptic curve.
    fn defining_equation(p: &[FieldElement<Self::BaseField>; 3]) -> FieldElement<Self::BaseField> {
        assert_ne!(Self::a(), FieldElement::zero()); // This could be a test
        assert_ne!(Self::d(), FieldElement::zero());
        assert_ne!(Self::a(), Self::d());

        let (x, y, z) = (&p[0], &p[1], &p[2]);
        Self::a() * x.pow(2_u16) + y.pow(2_u16)
            - z.pow(2_u16)
            - Self::d() * x.pow(2_u16) * y.pow(2_u16)
    }
}

/// Taken from moonmath manual page 97
#[derive(Debug, Clone)]
pub struct TinyJubJubEdwards;

impl IsEllipticCurve for TinyJubJubEdwards {
    type BaseField = U64PrimeField<13>;
    type PointRepresentation = EdwardsProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::create_point_from_affine(FieldElement::from(8), FieldElement::from(5))
    }

    fn create_point_from_affine(
        x: FieldElement<Self::BaseField>,
        y: FieldElement<Self::BaseField>,
    ) -> Self::PointRepresentation {
        Self::PointRepresentation::new([x, y, FieldElement::one()])
    }
}

impl IsEdwards for TinyJubJubEdwards {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(3)
    }

    fn d() -> FieldElement<Self::BaseField> {
        FieldElement::from(8)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::edwards::{element::EdwardsProjectivePoint, traits::TinyJubJubEdwards},
        field::element::FieldElement,
    };

    #[test]
    fn sum_works() {
        let p = EdwardsProjectivePoint::<TinyJubJubEdwards>::new([
            FieldElement::from(5),
            FieldElement::from(5),
            FieldElement::from(1),
        ]);
        let q = EdwardsProjectivePoint::<TinyJubJubEdwards>::new([
            FieldElement::from(8),
            FieldElement::from(5),
            FieldElement::from(1),
        ]);
        let expected = EdwardsProjectivePoint::<TinyJubJubEdwards>::new([
            FieldElement::from(0),
            FieldElement::from(1),
            FieldElement::from(1),
        ]);
        assert_eq!(p.operate_with(&q), expected);
    }
}
