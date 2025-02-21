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

    /// Returns the generator point of the TinyJubJub Montgomery curve.
    ///
    /// This generator is taken from **Moonmath Manual (page 91)**.
    ///
    /// # Safety
    ///
    /// - The generator coordinates `(3, 5, 1)` are **predefined** and are **valid** points
    ///   on the TinyJubJub Montgomery curve.
    /// - `unwrap_unchecked()` is used because the generator is **guaranteed** to satisfy
    ///   the Montgomery curve equation.
    /// - This function must **not** be modified unless the new generator is mathematically verified.
    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator point `(3, 5, 1)` is **mathematically verified**.
        // - `unwrap_unchecked()` is safe because the input values **guarantee** validity.

        Self::PointRepresentation::new([
            FieldElement::from(3),
            FieldElement::from(5),
            FieldElement::one(),
        ])
        .unwrap()
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
