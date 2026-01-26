use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;

/// Trait to add Edwards elliptic curve behavior to a struct.
///
/// # Completeness Requirement
///
/// This trait is designed for **complete** Edwards curves only. A twisted Edwards curve
/// `ax² + y² = 1 + dx²y²` is complete when:
/// - `a` is a square in the base field
/// - `d` is a non-square in the base field
///
/// For complete curves, the unified addition formula (Equation 5.38 in Moonmath, page 97)
/// has denominators that are never zero for any pair of points on the curve. This is
/// proven in Theorem 3.3 of <https://eprint.iacr.org/2007/286.pdf>.
///
/// **If you implement this trait for a non-complete curve, the addition operation may panic.**
///
/// # Example complete Edwards curves
/// - Ed25519 (a = -1, d = -121665/121666)
/// - JubJub (used in Zcash Sapling)
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
