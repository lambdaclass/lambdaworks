use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;

/// Trait to add elliptic curves behaviour to a struct.
/// We use the short Weierstrass form equation: `y^2 = x^3 + a * x  + b`.
pub trait IsShortWeierstrass: IsEllipticCurve + Clone + Debug {
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

pub trait Compress {
    type G1Point: IsGroup;
    type G2Point: IsGroup;
    type Error;

    #[cfg(feature = "alloc")]
    fn compress_g1_point(point: &Self::G1Point) -> alloc::vec::Vec<u8>;

    #[cfg(feature = "alloc")]
    fn compress_g2_point(point: &Self::G2Point) -> alloc::vec::Vec<u8>;

    fn decompress_g1_point(input_bytes: &mut [u8; 48]) -> Result<Self::G1Point, Self::Error>;

    fn decompress_g2_point(input_bytes: &mut [u8; 96]) -> Result<Self::G2Point, Self::Error>;
}
