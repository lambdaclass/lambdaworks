use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use std::fmt::Debug;

/// Trait to add elliptic curves behaviour to a struct.
pub trait IsEdwards: IsEllipticCurve + Clone + Debug {
    fn a() -> FieldElement<Self::BaseField>;

    fn d() -> FieldElement<Self::BaseField>;

    /// Evaluates the short Weierstrass equation at (x, y z).
    /// Used for checking if [x: y: z] belongs to the elliptic curve.
    fn defining_equation(p: &[FieldElement<Self::BaseField>; 3]) -> FieldElement<Self::BaseField> {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        (Self::a() * x.pow(2_u16) + y.pow(2_u16)) * z.pow(2_u16)
            - z.pow(4_u16)
            - Self::d() * x.pow(2_u16) * y.pow(2_u16)
    }
}
