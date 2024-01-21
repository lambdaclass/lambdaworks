use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;

macro_rules! define_short_weierstrass_trait {
    ($($bound:ident),*) => {
        /// Trait to add elliptic curves behaviour to a struct.
        /// We use the short Weierstrass form equation: `y^2 = x^3 + a * x  + b`.
        pub trait IsShortWeierstrass: $($bound +)* {
            /// `a` coefficient for the equation `y^2 = x^3 + a * x  + b`.
            fn a() -> FieldElement<Self::BaseField>;

            /// `b` coefficient for the equation  `y^2 = x^3 + a * x  + b`.
            fn b() -> FieldElement<Self::BaseField>;

            fn defining_equation(
                x: &FieldElement<Self::BaseField>,
                y: &FieldElement<Self::BaseField>,
            ) -> FieldElement<Self::BaseField> {
                y.pow(2_u16) - x.pow(3_u16) - Self::a() * x - Self::b()
            }
        }
    };
}

#[cfg(not(feature = "constant-time"))]
define_short_weierstrass_trait!(IsEllipticCurve, Clone, Debug);

#[cfg(feature = "constant-time")]
define_short_weierstrass_trait!(IsEllipticCurve, Clone, Debug, Copy);
