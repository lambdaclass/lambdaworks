use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;
/// Trait to add elliptic curves behaviour to a struct.
pub trait IsEdwards: IsEllipticCurve + Clone + Debug {
    fn a() -> FieldElement<Self::BaseField>;

    fn d() -> FieldElement<Self::BaseField>;

    // Edwards equation in affine coordinates:
    // ax^2 + y^2 - 1 = d * x^2 * y^2
    fn defining_equation(
        x: &FieldElement<Self::BaseField>,
        y: &FieldElement<Self::BaseField>,
    ) -> FieldElement<Self::BaseField> {
        let x2 = x.square();
        let y2 = y.square();
        (Self::a() * &x2 + &y2) - FieldElement::<Self::BaseField>::one() - Self::d() * x2 * y2
    }

    // Edwards equation in projective coordinates.
    // a * x^2 * z^2 + y^2 * z^2 - z^4 = d * x^2 * y^2
    fn defining_equation_projective(
        x: &FieldElement<Self::BaseField>,
        y: &FieldElement<Self::BaseField>,
        z: &FieldElement<Self::BaseField>,
    ) -> FieldElement<Self::BaseField> {
        let x2 = x.square();
        let y2 = y.square();
        let z2 = z.square();
        Self::a() * &x2 * &z2 + &y2 * &z2 - z2.square() - Self::d() * x2 * y2
    }
}
