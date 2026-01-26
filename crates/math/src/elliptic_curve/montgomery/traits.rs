use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;

/// Trait to add Montgomery elliptic curve behavior to a struct.
///
/// A Montgomery curve is defined by the equation: `by² = x³ + ax² + x`
///
/// # Requirements
///
/// For a valid Montgomery curve, the following must hold:
/// - `b ≠ 0` (required for the curve to be non-singular)
/// - `a² ≠ 4` (required for the curve to be non-singular)
///
/// The addition formula implementation relies on `b ≠ 0` for correctness.
/// If `b = 0`, division operations in point addition will fail.
pub trait IsMontgomery: IsEllipticCurve + Clone + Debug {
    /// Returns the coefficient `a` of the curve equation `by² = x³ + ax² + x`.
    fn a() -> FieldElement<Self::BaseField>;

    /// Returns the coefficient `b` of the curve equation `by² = x³ + ax² + x`.
    ///
    /// **Important**: This must be non-zero for a valid Montgomery curve.
    fn b() -> FieldElement<Self::BaseField>;

    /// Evaluates the equation at (x, y).
    /// Used for checking if the point belongs to the elliptic curve.
    /// Equation: by^2 = x^3 + ax^2 + x.
    fn defining_equation(
        x: &FieldElement<Self::BaseField>,
        y: &FieldElement<Self::BaseField>,
    ) -> FieldElement<Self::BaseField> {
        (Self::b() * y.square()) - (x.pow(3_u16) + Self::a() * x.square() + x)
    }

    /// Evaluates the equation at the projective point (x, y, z).
    /// Projective equation: zby^2 = x^3 + zax^2 + z^2x
    fn defining_equation_projective(
        x: &FieldElement<Self::BaseField>,
        y: &FieldElement<Self::BaseField>,
        z: &FieldElement<Self::BaseField>,
    ) -> FieldElement<Self::BaseField> {
        z * Self::b() * y.square() - x.pow(3_u16) - z * Self::a() * x.square() - z.square() * x
    }
}
