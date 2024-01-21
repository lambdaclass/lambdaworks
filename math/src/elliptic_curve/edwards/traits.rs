use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;
macro_rules! define_edwards_trait {
    ($($bound:ident),*) => {
        /// Trait to add elliptic curves behaviour to a struct.
        pub trait IsEdwards: $($bound +)* {
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
    };
}

#[cfg(not(feature = "constant-time"))]
define_edwards_trait!(IsEllipticCurve, Clone, Debug);

#[cfg(feature = "constant-time")]
define_edwards_trait!(IsEllipticCurve, Clone, Debug, Copy);
