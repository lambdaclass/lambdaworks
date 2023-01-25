use crate::field::field_element::{FieldElement, FieldOperations};

use super::cyclic_group::CyclicBilinearGroup;
use std::marker::PhantomData;
use std::ops;
use std::fmt::Debug;


pub trait EllipticCurveOperations: Clone + Debug {
    type BaseField: Clone + Debug + FieldOperations;

    fn a() -> FieldElement<Self::BaseField>;
    fn b() -> FieldElement<Self::BaseField>;
    fn generator_affine_x() -> FieldElement<Self::BaseField>;
    fn generator_affine_y() -> FieldElement<Self::BaseField>;
    fn embedding_degree() -> u32;
    fn order_r() -> u64;
    fn order_p() -> u64;

    fn order_field_extension() -> u64 {
        Self::order_p().pow(Self::embedding_degree())
    }

    fn target_normalization_power() -> u64 {
        (Self::order_field_extension() - 1) / Self::order_r()
    }

    /// Evaluates the short Weierstrass equation at (x, y z).
    /// Useful for checking if (x, y, z) belongs to the elliptic curve.
    fn defining_equation(p: &[&FieldElement<Self::BaseField>; 3]) -> FieldElement<Self::BaseField> {
        let (x, y, z) = (p[0], p[1], p[2]);
        y.pow(2) * z - x.pow(3) - Self::a() * x * z.pow(2) - Self::b() * z.pow(3)
    }

    fn neutral_element() -> [FieldElement<Self::BaseField>; 3] {
        [FieldElement::zero(), FieldElement::one(), FieldElement::zero()]
    }

    fn is_neutral_element(p: &[&FieldElement<Self::BaseField>; 3]) -> bool {
        let (x, y, z) = (p[0], p[1], p[2]);
        x == &FieldElement::zero() && y == &FieldElement::one() && z == &FieldElement::zero()
    }

    /// Normalize the projective coordinates to obtain affine coordinates
    /// of the form (x, y, 1)
    /// Panics if `self` is the point at infinity
    fn affine(p: &[&FieldElement<Self::BaseField>; 3]) -> [FieldElement<Self::BaseField>; 3] {
        assert!(!Self::is_neutral_element(p), "The point at infinity is not affine.");
        let (x, y, z) = (p[0], p[1], p[2]);
        [x / z, y / z, FieldElement::one()]
    }

    fn add(
        p: &[&FieldElement<Self::BaseField>; 3], 
        q: &[&FieldElement<Self::BaseField>; 3]
    ) -> [FieldElement<Self::BaseField>; 3] {
        let (px, py, pz) = (p[0], p[1], p[2]);
        let (qx, qy, qz) = (q[0], q[1], q[2]);
        if Self::is_neutral_element(q) {
            [px.clone(), py.clone(), pz.clone()]
        } else if Self::is_neutral_element(p) {
            [qx.clone(), qy.clone(), qz.clone()]
        } else {
            let u1 = qy * pz;
            let u2 = py * qz;
            let v1 = qx * pz;
            let v2 = px * qz;
            if v1 == v2 {
                if u1 != u2 || *py == FieldElement::from(0) {
                    Self::neutral_element()
                } else {
                    let w = Self::a() * pz.pow(2)
                        + FieldElement::from(3) * px.pow(2);
                    let s = py * pz;
                    let b = px * py * &s;
                    let h = w.pow(2) - FieldElement::from(8) * &b;
                    let xp = FieldElement::from(2) * &h * &s;
                    let yp = w * (FieldElement::from(4) * &b - &h)
                        - FieldElement::from(8) * py.pow(2) * s.pow(2);
                    let zp = FieldElement::from(8) * s.pow(3);
                    [xp, yp, zp]
                }
            } else {
                let u = u1 - &u2;
                let v = v1 - &v2;
                let w = pz * qz;
                let a = u.pow(2) * &w - v.pow(3) - FieldElement::from(2) * v.pow(2) * &v2;
                let xp = &v * &a;
                let yp = u * (v.pow(2) * v2 - a) - v.pow(3) * u2;
                let zp = v.pow(3) * w;
                [xp, yp, zp]
            }
        }
    }

}

/// Represents an elliptic curve point using the projective short Weierstrass form:
///   y^2 * z = x^3 + a * x * z^2 + b * z^3
/// x, y and z variables are field extension elements.
#[derive(Debug, Clone)]
pub struct EllipticCurveElement<E: EllipticCurveOperations> {
    value: [FieldElement<E::BaseField>; 3],
    elliptic_curve: PhantomData<E>
}

impl<E: EllipticCurveOperations> EllipticCurveElement<E> {
    fn x(&self) -> &FieldElement<E::BaseField>{
        &self.value[0]
    }

    fn y(&self) -> &FieldElement<E::BaseField>{
        &self.value[1]
    }

    fn z(&self) -> &FieldElement<E::BaseField>{
        &self.value[2]
    }

    fn to_affine(&self) -> Self {
        Self {
            value: E::affine(&[&self.value[0], &self.value[1], &self.value[2]]),
            elliptic_curve: PhantomData
        }
    }
}

impl<E: EllipticCurveOperations> EllipticCurveElement<E> 
where EllipticCurveElement<E>: CyclicBilinearGroup
{
    /// Creates an elliptic curve point giving the (x, y, z) coordinates.
    fn new(x: FieldElement<E::BaseField>,
           y: FieldElement<E::BaseField>,
           z: FieldElement<E::BaseField>) -> Self {
        assert_eq!(
            E::defining_equation(&[&x, &y, &z]),
            FieldElement::zero(),
            "Point ({:?}, {:?}, {:?}) does not belong to the elliptic curve.",
            x,
            y,
            z
        );
        Self { value: [x, y, z], elliptic_curve: PhantomData }
    }

    /// Evaluates the line between points `self` and `r` at point `q`
    fn line(&self, r: &Self, q: &Self) -> FieldElement<E::BaseField> {
        assert_ne!(
            *q,
            Self::neutral_element(),
            "q cannot be the point at infinity."
        );
        if *self == Self::neutral_element() || *r == Self::neutral_element() {
            if self == r {
                return FieldElement::one();
            }
            if *self == Self::neutral_element() {
                q.x() - r.x()
            } else {
                q.x() - self.x()
            }
        } else if self != r {
            if self.x() == r.x() {
                return q.x() - self.x();
            } else {
                let l = (r.y() - self.y()) / (r.x() - self.x());
                return q.y() - self.y() - l * (q.x() - self.x());
            }
        } else {
            let numerator = FieldElement::from(3) * &self.x().pow(2)
                + E::a();
            let denominator = FieldElement::from(2) * self.y();
            if denominator == FieldElement::from(0) {
                return q.x() - self.x();
            } else {
                let l = numerator / denominator;
                return q.y() - self.y() - l * (q.x() - self.x());
            }
        }
    }

    /// Computes Miller's algorithm between points `p` and `q`.
    /// The implementaiton is based on Sagemath's sourcecode:
    /// See `_miller_` method on page 114
    /// https://www.sagemath.org/files/thesis/hansen-thesis-2009.pdf
    /// Other resources can be found at "Pairings for beginners" from Craig Costello, Algorithm 5.1, page 79.
    fn miller(p: &Self, q: &Self) -> FieldElement<E::BaseField> {
        let p = p.to_affine();
        let q = q.to_affine();
        let mut order_r = E::order_r();
        let mut bs = vec![];
        while order_r > 0 {
            bs.insert(0, order_r & 1);
            order_r >>= 1;
        }

        let mut f = FieldElement::from(1);
        let mut r = p.clone();

        for b in bs[1..].iter() {
            let s = r.operate_with(&r).to_affine();
            f = f.pow(2) * (r.line(&r, &q) / s.line(&-(&s), &q));
            r = s;

            if *b == 1 {
                let mut s = r.operate_with(&p);
                if s != Self::neutral_element() {
                    s = s.to_affine();
                }
                f = f * (r.line(&p, &q) / s.line(&-(&s), &q));
                r = s;
            }
        }
        f
    }

    /// Computes the Weil pairing between points `p` and `q`.
    /// See "Pairing for beginners" from Craig Costello, page 79.
    #[allow(unused)]
    fn weil_pairing(p: &Self, q: &Self) -> FieldElement<E::BaseField> {
        if *p == Self::neutral_element() || *q == Self::neutral_element() || p == q {
            FieldElement::from(1)
        } else {
            let numerator = Self::miller(p, q);
            let denominator = Self::miller(q, p);
            let result = numerator / denominator;
            -result
        }
    }

    /// Computes the Tate pairing between points `p` and `q`.
    /// See "Pairing for beginners" from Craig Costello, page 79.
    fn tate_pairing(p: &Self, q: &Self) -> FieldElement<E::BaseField> {
        if *p == Self::neutral_element() || *q == Self::neutral_element() || p == q {
            FieldElement::from(1)
        } else {
            Self::miller(p, q).pow(E::target_normalization_power() as u128)
        }
    }

    /// Apply a distorsion map to point `p`.
    /// This is useful for converting points living in the base field
    /// to points living in the extension field.
    /// The current implementation only works for the elliptic curve with A=1 and B=0
    /// ORDER_P=59. This curve was chosen because it is supersingular.
    fn distorsion_map(p: &Self) -> Self {
        todo!()
        //let t = FieldElement::new([FieldElement::zero(), FieldElement::one()]);
        //Self::new(-&p.x, &p.y * t, p.z.clone())
    }
}

impl<E: EllipticCurveOperations> PartialEq for EllipticCurveElement<E> {
    fn eq(&self, other: &Self) -> bool {
        // Projective equality relation: first point has to be a multiple of the other
        (self.x() * other.z() == self.z() * other.x()) && (self.x() * other.y() == self.y() * other.x())
    }
}
impl<E: EllipticCurveOperations> Eq for EllipticCurveElement<E> {}

impl<E: EllipticCurveOperations> ops::Neg for &EllipticCurveElement<E> {
    type Output = EllipticCurveElement<E>;

    fn neg(self) -> Self::Output {
        Self::Output::new(self.x().clone(), -self.y().clone(), self.z().clone())
    }
}

impl<E: EllipticCurveOperations> ops::Neg for EllipticCurveElement<E> {
    type Output = EllipticCurveElement<E>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<E: EllipticCurveOperations> CyclicBilinearGroup for EllipticCurveElement<E> {
    type PairingOutput = FieldElement<E::BaseField>;

    fn generator() -> Self {
        Self::new(
            E::generator_affine_x(),
            E::generator_affine_y(),
            FieldElement::from(1),
        )
    }

    fn neutral_element() -> Self {
        Self::new(
            FieldElement::from(0),
            FieldElement::from(1),
            FieldElement::from(0),
        )
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from Moonmath (Algorithm 7, page 89)
    fn operate_with(&self, other: &Self) -> Self {
        if *other == Self::neutral_element() {
            self.clone()
        } else if *self == Self::neutral_element() {
            other.clone()
        } else {
            let u1 = other.y() * self.z();
            let u2 = self.y() * other.z();
            let v1 = other.x() * self.z();
            let v2 = self.x() * other.z();
            if v1 == v2 {
                if u1 != u2 || *self.y() == FieldElement::from(0) {
                    Self::neutral_element()
                } else {
                    let w = E::a() * self.z().pow(2)
                        + FieldElement::from(3) * self.x().pow(2);
                    let s = self.y() * self.z();
                    let b = self.x() * self.y() * &s;
                    let h = w.pow(2) - FieldElement::from(8) * &b;
                    let xp = FieldElement::from(2) * &h * &s;
                    let yp = w * (FieldElement::from(4) * &b - &h)
                        - FieldElement::from(8) * self.y().pow(2) * s.pow(2);
                    let zp = FieldElement::from(8) * s.pow(3);
                    Self::new(xp, yp, zp)
                }
            } else {
                let u = u1 - &u2;
                let v = v1 - &v2;
                let w = self.z() * other.z();
                let a = u.pow(2) * &w - v.pow(3) - FieldElement::from(2) * v.pow(2) * &v2;
                let xp = &v * &a;
                let yp = u * (v.pow(2) * v2 - a) - v.pow(3) * u2;
                let zp = v.pow(3) * w;
                Self::new(xp, yp, zp)
            }
        }
    }

    /// Computes a Type 1 Tate pairing between `self` and `other.
    /// See "Pairing for beginners" from Craig Costello, section 4.2 Pairing types, page 58.
    /// Note that a distorsion map is applied to `other` before using the Tate pairing.
    /// So this method can be called with two field extension elements from the base field.
    fn pairing(&self, other: &Self) -> Self::PairingOutput {
        //Self::tate_pairing(self, &Self::distorsion_map(other))
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{field::{quadratic_extension::{HasQuadraticNonResidue, QuadraticExtensionFieldElement, QuadraticExtensionField}, u64_prime_field::{U64PrimeField, U64FieldElement}}, config::{ORDER_P, ORDER_R}};

    use super::*;

    #[derive(Debug, Clone)]
    pub struct QuadraticNonResidue;
    impl HasQuadraticNonResidue<U64PrimeField<ORDER_P>> for QuadraticNonResidue {
        fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
            -FieldElement::one()
        }
    }

    #[allow(clippy::upper_case_acronyms)]
    type FEE = QuadraticExtensionFieldElement<U64PrimeField<ORDER_P>, QuadraticNonResidue>;

    #[derive(Clone, Debug)]
    pub struct CurrentCurve;
    impl EllipticCurveOperations for CurrentCurve {
        type BaseField = QuadraticExtensionField<U64PrimeField<ORDER_P>, QuadraticNonResidue>;
        
        fn a() -> FieldElement<Self::BaseField> {
            FieldElement::from(1)
        }

        fn b() -> FieldElement<Self::BaseField>  {
            FieldElement::from(0)
        }

        fn generator_affine_x() -> FieldElement<Self::BaseField> {
            FieldElement::from(35)
        }

        fn generator_affine_y() -> FieldElement<Self::BaseField>  {
            FieldElement::from(31)
        }

        fn embedding_degree() -> u32 {
            2
        }

        fn order_r() -> u64 {
            5
        }

        fn order_p() -> u64 {
            59
        }
    }

    // This tests only apply for the specific curve found in the configuration file.
    #[test]
    fn create_valid_point_works() {
        let point = EllipticCurveElement::<CurrentCurve>::new(
            FEE::from(35),
            FEE::from(31),
            FEE::from(1),
        );
        assert_eq!(*point.x(), FEE::new_base(35));
        assert_eq!(*point.y(), FEE::new_base(31));
        assert_eq!(*point.z(), FEE::new_base(1));
    }

    #[test]
    #[should_panic]
    fn create_invalid_points_panicks() {
        EllipticCurveElement::<CurrentCurve>::new(
            FEE::new_base(0),
            FEE::new_base(1),
            FEE::new_base(1),
        );
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = EllipticCurveElement::<CurrentCurve>::generator();
        assert_eq!(g.operate_with(&g).operate_with(&g), g.operate_with_self(3));
    }

    #[test]
    fn operate_with_self_works_2() {
        let mut point_1 = EllipticCurveElement::<CurrentCurve>::generator();
        point_1 = point_1.operate_with_self(ORDER_R as u128);
        assert_eq!(point_1, EllipticCurveElement::<CurrentCurve>::neutral_element());
    }

    #[test]
    fn doubling_a_point_works() {
        let point = EllipticCurveElement::<CurrentCurve>::new(
            FEE::new_base(35),
            FEE::new_base(31),
            FEE::new_base(1),
        );
        let expected_result = EllipticCurveElement::<CurrentCurve>::new(
            FEE::new_base(25),
            FEE::new_base(29),
            FEE::new_base(1),
        );
        assert_eq!(point.operate_with_self(2).to_affine(), expected_result);
    }

    #[test]
    fn test_weil_pairing() {
        type FE = U64FieldElement<ORDER_P>;
        let pa = EllipticCurveElement::<CurrentCurve>::new(
            FEE::new_base(35),
            FEE::new_base(31),
            FEE::new_base(1),
        );
        let pb = EllipticCurveElement::<CurrentCurve>::new(
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
            FEE::new_base(1),
        );
        let expected_result = FEE::new([FE::new(46), FE::new(3)]);

        let result_weil = EllipticCurveElement::<CurrentCurve>::weil_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }

    #[test]
    fn test_tate_pairing() {
        type FE = U64FieldElement<ORDER_P>;
        let pa = EllipticCurveElement::<CurrentCurve>::new(
            FEE::new_base(35),
            FEE::new_base(31),
            FEE::new_base(1),
        );
        let pb = EllipticCurveElement::<CurrentCurve>::new(
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
            FEE::new_base(1),
        );
        let expected_result = FEE::new([FE::new(42), FE::new(19)]);

        let result_weil = EllipticCurveElement::<CurrentCurve>::tate_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }
}
