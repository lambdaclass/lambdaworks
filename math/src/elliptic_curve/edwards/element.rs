use crate::elliptic_curve::projective_point::ProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use crate::field::fields::u64_prime_field::U64PrimeField;
use crate::unsigned_integer::traits::IsUnsignedInteger;
use std::fmt::Debug;

/// Trait to add elliptic curves behaviour to a struct.
pub trait IsEdwards: IsEllipticCurve + Clone + Debug {
    /// The type used to store order_p and order_r.
    type UIntOrders: IsUnsignedInteger;

    fn a() -> FieldElement<Self::BaseField>;

    fn d() -> FieldElement<Self::BaseField>;

    /// Order of the subgroup of the curve (e.g.: number of elements in
    /// the subgroup of the curve).
    fn order_r() -> Self::UIntOrders;

    /// Order of the base field (e.g.: order of the field where `a` and `b` are defined).
    fn order_p() -> Self::UIntOrders;

    /// Evaluates the short Weierstrass equation at (x, y z).
    /// Used for checking if [x: y: z] belongs to the elliptic curve.
    fn defining_equation(p: &[FieldElement<Self::BaseField>; 3]) -> FieldElement<Self::BaseField> {
        assert_ne!(Self::a(), FieldElement::zero()); // This could be a test
        assert_ne!(Self::d(), FieldElement::zero());
        assert_ne!(Self::a(), Self::d());

        let (x, y, z) = (&p[0], &p[1], &p[2]);
        Self::a() * x.pow(2_u16) + y.pow(2_u16) - z.pow(2_u16) - Self::d() * x.pow(2_u16) * y.pow(2_u16)
    }

    /// Projective equality relation: `p` has to be a multiple of `q`
    fn eq(p: &[FieldElement<Self::BaseField>; 3], q: &[FieldElement<Self::BaseField>; 3]) -> bool {
        let (px, py, pz) = (&p[0], &p[1], &p[2]);
        let (qx, qy, qz) = (&q[0], &q[1], &q[2]);
        (px * qz == pz * qx) && (px * qy == py * qx)
    }

    /// The point at infinity.
    fn neutral_element() -> [FieldElement<Self::BaseField>; 3] {
        [
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ]
    }

    /// Check if a projective point `p` is the point at inifinity.
    fn is_neutral_element(p: &[FieldElement<Self::BaseField>; 3]) -> bool {
        Self::eq(p, &Self::neutral_element())
    }

    /// Returns the normalized projective coordinates to obtain "affine" coordinates
    /// of the form [x: y: 1]
    /// Panics if `self` is the point at infinity
    fn affine(p: &[FieldElement<Self::BaseField>; 3]) -> [FieldElement<Self::BaseField>; 3] {
        assert!(
            !Self::is_neutral_element(p),
            "The point at infinity is not affine."
        );
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        [x / z, y / z, FieldElement::one()]
    }

    /// Returns the sum of projective points `p` and `q`
    /// Taken from "Moonmath" (Algorithm 7, page 89)
    fn add_edwards(
        p: &[FieldElement<Self::BaseField>; 3],
        q: &[FieldElement<Self::BaseField>; 3],
    ) -> [FieldElement<Self::BaseField>; 3] {
        // TODO: Remove repeated operations
        let [x1, y1, _] = Self::affine(p);
        let [x2, y2, _] = Self::affine(q);
        
        let num_s1 = &x1 * &y2 + &y1 * &x2;
        let den_s1 = FieldElement::one() + Self::d() * &x1 * &x2 * &y1 * &y2;

        let num_s2 = &y1 * &y2 - Self::a() * &x1 * &x2;
        let den_s2 = FieldElement::one() - Self::d() * &x1 * &x2 * &y1 * &y2;

        [num_s1 / den_s1, num_s2 / den_s2, FieldElement::one()]
    }

    /// Returns the additive inverse of the projective point `p`
    fn neg(p: &[FieldElement<Self::BaseField>; 3]) -> [FieldElement<Self::BaseField>; 3] {
        todo!()
    }
}


/// Taken from moonmath manual page 97
#[derive(Debug, Clone)]
pub struct TinyJubJubEdwards;

impl IsEllipticCurve for TinyJubJubEdwards {
    type BaseField = U64PrimeField<13>;
    type PointRepresentation = ProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        todo!()
    }

    fn create_affine_point(
        x: FieldElement<Self::BaseField>,
        y: FieldElement<Self::BaseField>,
    ) -> Self::PointRepresentation {
        todo!()
    }

    fn add(
        p: &Self::PointRepresentation,
        q: &Self::PointRepresentation,
    ) -> Self::PointRepresentation {
        Self::PointRepresentation::new(Self::add_edwards(p.coordinates(), q.coordinates()))
    }
}

impl IsEdwards for TinyJubJubEdwards {
    type UIntOrders = u64;

    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(3)
    }

    fn d() -> FieldElement<Self::BaseField> {
        FieldElement::from(8)
    }

    fn order_r() -> Self::UIntOrders {
        todo!()
    }

    fn order_p() -> Self::UIntOrders {
        13
    }
}


#[cfg(test)]
mod tests {
    use crate::{elliptic_curve::{edwards::element::TinyJubJubEdwards, projective_point::ProjectivePoint}, field::element::FieldElement, cyclic_group::IsGroup};

    #[test]
    fn sum_works() {
        let p = ProjectivePoint::<TinyJubJubEdwards>::new([FieldElement::from(5), FieldElement::from(5), FieldElement::from(1)]);
        let q = ProjectivePoint::<TinyJubJubEdwards>::new([FieldElement::from(8), FieldElement::from(5), FieldElement::from(1)]);
        let expected = ProjectivePoint::<TinyJubJubEdwards>::new([FieldElement::from(0), FieldElement::from(1), FieldElement::from(1)]);
        assert_eq!(p.operate_with(&q), expected);
    }
}
