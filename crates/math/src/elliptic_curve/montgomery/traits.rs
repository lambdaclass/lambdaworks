use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;

/// Trait to add elliptic curves behaviour to a struct.
pub trait IsMontgomery: IsEllipticCurve + Clone + Debug {
    fn a() -> FieldElement<Self::BaseField>;

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
