use crate::{
    elliptic_curve::{
        montgomery::{point::MontgomeryProjectivePoint, traits::IsMontgomery},
        traits::IsEllipticCurve,
    },
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
};

/// Taken from moonmath manual page 91
#[derive(Debug, Clone)]
pub struct TinyJubJubMontgomery;

impl IsEllipticCurve for TinyJubJubMontgomery {
    type BaseField = U64PrimeField<13>;
    type PointRepresentation = MontgomeryProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::from(3),
            FieldElement::from(5),
            FieldElement::one(),
        ])
    }
}

impl IsMontgomery for TinyJubJubMontgomery {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(6)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(7)
    }
}
