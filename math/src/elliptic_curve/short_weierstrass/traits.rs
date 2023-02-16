use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use crate::unsigned_integer::traits::IsUnsignedInteger;
use std::fmt::Debug;

/// Trait to add elliptic curves behaviour to a struct.
/// We use the short Weierstrass form equation: `y^2 = x^3 + a * x  + b`.
pub trait IsShortWeierstrass: IsEllipticCurve + Clone + Debug {
    /// The type used to store order_p and order_r.
    type UIntOrders: IsUnsignedInteger;

    /// `a` coefficient for the equation `y^2 = x^3 + a * x  + b`.
    fn a() -> FieldElement<Self::BaseField>;

    /// `b` coefficient for the equation  `y^2 = x^3 + a * x  + b`.
    fn b() -> FieldElement<Self::BaseField>;

    /// Order of the subgroup of the curve (e.g.: number of elements in
    /// the subgroup of the curve).
    fn order_r() -> Self::UIntOrders;

    /// Order of the base field (e.g.: order of the field where `a` and `b` are defined).
    fn order_p() -> Self::UIntOrders;

    /// The big-endian bit representation of the normalization power for the Tate pairing.
    /// This is computed as:
    ///  (order_p.pow(embedding_degree) - 1) / order_r
    /// TODO: This is only used for the Tate pairing and will disappear. Something ideas like
    /// the ones on this paper (https://eprint.iacr.org/2020/875.pdf) will be implemented.
    fn target_normalization_power() -> Vec<u64>;

    /// Evaluates the short Weierstrass equation at (x, y z).
    /// Used for checking if [x: y: z] belongs to the elliptic curve.
    fn defining_equation(p: &[FieldElement<Self::BaseField>; 3]) -> FieldElement<Self::BaseField> {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        y.pow(2_u16) * z - x.pow(3_u16) - Self::a() * x * z.pow(2_u16) - Self::b() * z.pow(3_u16)
    }

    /// Projective equality relation: `p` has to be a multiple of `q`
    fn eq(p: &[FieldElement<Self::BaseField>; 3], q: &[FieldElement<Self::BaseField>; 3]) -> bool {
        let (px, py, pz) = (&p[0], &p[1], &p[2]);
        let (qx, qy, qz) = (&q[0], &q[1], &q[2]);
        (px * qz == pz * qx) && (px * qy == py * qx)
    }
}

/// Trait to add distortion maps to Elliptic Curves.
/// Typically used to support type I pairings.
/// For more info look into page 56 of "Pairings for beginners".
pub trait HasDistortionMap: IsShortWeierstrass {
    fn distorsion_map(p: &[FieldElement<Self::BaseField>; 3])
        -> [FieldElement<Self::BaseField>; 3];
}
