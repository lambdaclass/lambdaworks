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

    /// Evaluates the short Weierstrass equation at (x, y z).
    /// Used for checking if [x: y: z] belongs to the elliptic curve.
    fn defining_equation(p: &[FieldElement<Self::BaseField>; 3]) -> FieldElement<Self::BaseField> {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        y.pow(2_u16) * z - x.pow(3_u16) - Self::a() * x * z.pow(2_u16) - Self::b() * z.pow(3_u16)
    }
}

/// Trait to add distortion maps to Elliptic Curves.
/// Typically used to support type I pairings.
/// For more info look into page 56 of "Pairings for beginners".
pub trait HasDistortionMap: IsShortWeierstrass {
    fn distorsion_map(p: &[FieldElement<Self::BaseField>; 3])
        -> [FieldElement<Self::BaseField>; 3];
}
