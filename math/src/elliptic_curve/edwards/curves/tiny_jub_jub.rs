use crate::{
    elliptic_curve::{
        edwards::{point::EdwardsProjectivePoint, traits::IsEdwards},
        traits::{EllipticCurveError, IsEllipticCurve},
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
        Self::PointRepresentation::new([
            FieldElement::from(8),
            FieldElement::from(5),
            FieldElement::one(),
        ])
    }

    fn create_point_from_affine(
        x: FieldElement<Self::BaseField>,
        y: FieldElement<Self::BaseField>,
    ) -> Result<Self::PointRepresentation, EllipticCurveError> {
        let coordinates = [x, y, FieldElement::one()];
        if Self::defining_equation(&coordinates) != FieldElement::zero() {
            Err(EllipticCurveError::InvalidPoint)
        } else {
            Ok(Self::PointRepresentation::new(coordinates))
        }
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
