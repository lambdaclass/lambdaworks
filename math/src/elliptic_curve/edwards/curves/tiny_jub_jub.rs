use crate::{
    elliptic_curve::{
        edwards::{point::EdwardsProjectivePoint, traits::IsEdwards},
        traits::IsEllipticCurve,
    },
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
};

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
