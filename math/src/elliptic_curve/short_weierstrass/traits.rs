use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use std::fmt::Debug;

/// Trait to add elliptic curves behaviour to a struct.
/// We use the short Weierstrass form equation: `y^2 = x^3 + a * x  + b`.
pub trait IsShortWeierstrass: IsEllipticCurve + Clone + Debug {
    /// `a` coefficient for the equation `y^2 = x^3 + a * x  + b`.
    fn a() -> FieldElement<Self::BaseField>;

    /// `b` coefficient for the equation  `y^2 = x^3 + a * x  + b`.
    fn b() -> FieldElement<Self::BaseField>;

    fn defining_equation(p: &[FieldElement<Self::BaseField>; 3]) -> FieldElement<Self::BaseField> {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        y.pow(2_u16) * z - x.pow(3_u16) - Self::a() * x * z.pow(2_u16) - Self::b() * z.pow(3_u16)
    }
}
