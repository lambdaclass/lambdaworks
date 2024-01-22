use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;

#[cfg(not(feature = "constant-time"))]
pub trait IsEdwards: IsEllipticCurve + Clone + Debug {
    fn a() -> FieldElement<Self::BaseField>;

    fn d() -> FieldElement<Self::BaseField>;

    fn defining_equation(
        x: &FieldElement<Self::BaseField>,
        y: &FieldElement<Self::BaseField>,
    ) -> FieldElement<Self::BaseField> {
        (Self::a() * x.pow(2_u16) + y.pow(2_u16))
            - FieldElement::<Self::BaseField>::one()
            - Self::d() * x.pow(2_u16) * y.pow(2_u16)
    }
}

#[cfg(feature = "constant-time")]
pub trait IsEdwards: IsEllipticCurve + Clone + Copy + Debug {
    fn a() -> FieldElement<Self::BaseField>;

    fn d() -> FieldElement<Self::BaseField>;

    fn defining_equation(
        x: &FieldElement<Self::BaseField>,
        y: &FieldElement<Self::BaseField>,
    ) -> FieldElement<Self::BaseField> {
        (Self::a() * x.pow(2_u16) + y.pow(2_u16))
            - FieldElement::<Self::BaseField>::one()
            - Self::d() * x.pow(2_u16) * y.pow(2_u16)
    }
}
