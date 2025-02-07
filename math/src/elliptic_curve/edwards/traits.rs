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
        (Self::a() * x.pow(2_u16) + y.pow(2_u16))
            - FieldElement::<Self::BaseField>::one()
            - Self::d() * x.pow(2_u16) * y.pow(2_u16)
    }

    // Edwards equation in projective coordinates.
    // a * x^2 * z^2 + y^2 * z^2 - z^4 = d * x^2 * y^2
    fn defining_equation_projective(
        x: &FieldElement<Self::BaseField>,
        y: &FieldElement<Self::BaseField>,
        z: &FieldElement<Self::BaseField>,
    ) -> FieldElement<Self::BaseField> {
        Self::a() * x.square() * z.square() + y.square() * z.square()
            - z.square().square()
            - Self::d() * x.square() * y.square()
    }
}
