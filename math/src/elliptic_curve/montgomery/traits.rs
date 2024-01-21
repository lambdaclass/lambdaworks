use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;

macro_rules! define_montgomery_trait {
    ($($bound:ident),*) => {
        /// Trait to add elliptic curves behaviour to a struct.
        pub trait IsMontgomery: $($bound +)* {
            fn a() -> FieldElement<Self::BaseField>;

            fn b() -> FieldElement<Self::BaseField>;

            /// Evaluates the short Weierstrass equation at (x, y z).
            /// Used for checking if [x: y: z] belongs to the elliptic curve.
            fn defining_equation(
                x: &FieldElement<Self::BaseField>,
                y: &FieldElement<Self::BaseField>,
            ) -> FieldElement<Self::BaseField> {
                (Self::b() * y.pow(2_u16)) - (x.pow(3_u16) + Self::a() * x.pow(2_u16) + x)
            }
        }
    };
}

#[cfg(not(feature = "constant-time"))]
define_montgomery_trait!(IsEllipticCurve, Clone, Debug);

#[cfg(feature = "constant-time")]
define_montgomery_trait!(IsEllipticCurve, Clone, Debug, Copy);
