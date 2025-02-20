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

    /// Returns the generator point of the TinyJubJub Edwards curve.
    ///
    /// This generator is taken from **Moonmath Manual (page 97)**.
    ///
    /// # Safety
    ///
    /// - The generator coordinates `(8, 5, 1)` are **predefined** and belong to the TinyJubJub curve.
    /// - `unwrap_unchecked()` is used because the generator is a **verified valid point**,
    ///   meaning there is **no risk** of runtime failure.
    /// - This function must **not** be modified unless the new generator is mathematically verified.
    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator point `(8, 5, 1)` is **mathematically valid** on the curve.
        // - `unwrap_unchecked()` is safe because we **know** the point satisfies the curve equation.
        unsafe {
            Self::PointRepresentation::new([
                FieldElement::from(8),
                FieldElement::from(5),
                FieldElement::one(),
            ])
            .unwrap_unchecked()
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
