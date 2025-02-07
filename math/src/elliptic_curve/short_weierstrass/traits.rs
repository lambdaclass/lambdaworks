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
        y.square() - ((x.square() + Self::a()) * x + Self::b())
    }

    // Evaluates the projective equation:
    // y^2 * z = x^3 + a * x * z^2 + b * z^3
    fn defining_equation_projective(
        x: &FieldElement<Self::BaseField>,
        y: &FieldElement<Self::BaseField>,
        z: &FieldElement<Self::BaseField>,
    ) -> FieldElement<Self::BaseField> {
        y.square() * z - ((x.square() + Self::a() * z.square()) * x + Self::b() * z.square() * z)
    }
}

pub trait Compress {
    type G1Point: IsGroup;
    type G2Point: IsGroup;
    type G1Compressed;
    type G2Compressed;
    type Error;

    #[cfg(feature = "alloc")]
    fn compress_g1_point(point: &Self::G1Point) -> Self::G1Compressed;

    #[cfg(feature = "alloc")]
    fn compress_g2_point(point: &Self::G2Point) -> Self::G2Compressed;

    fn decompress_g1_point(input_bytes: &mut [u8]) -> Result<Self::G1Point, Self::Error>;

    fn decompress_g2_point(input_bytes: &mut [u8]) -> Result<Self::G2Point, Self::Error>;
}
