use crate::field::field_element::{FieldElement, HasFieldOperations};

use super::cyclic_group::CyclicBilinearGroup;
use std::fmt::Debug;
use std::marker::PhantomData;

pub trait HasEllipticCurveOperations: Clone + Debug {
    type BaseField: HasFieldOperations + Clone + Debug;

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
    fn defining_equation(p: &[FieldElement<Self::BaseField>; 3]) -> FieldElement<Self::BaseField> {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        y.pow(2) * z - x.pow(3) - Self::a() * x * z.pow(2) - Self::b() * z.pow(3)
    }

    /// Projective equality relation: `p` has to be a multiple of `q`
    fn eq(p: &[FieldElement<Self::BaseField>; 3], q: &[FieldElement<Self::BaseField>; 3]) -> bool {
        let (px, py, pz) = (&p[0], &p[1], &p[2]);
        let (qx, qy, qz) = (&q[0], &q[1], &q[2]);
        (px * qz == pz * qx) && (px * qy == py * qx)
    }

    fn neutral_element() -> [FieldElement<Self::BaseField>; 3] {
        [
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ]
    }

    fn is_neutral_element(p: &[FieldElement<Self::BaseField>; 3]) -> bool {
        Self::eq(p, &Self::neutral_element())
    }

    /// Normalize the projective coordinates to obtain affine coordinates
    /// of the form (x, y, 1)
    /// Panics if `self` is the point at infinity
    fn affine(p: &[FieldElement<Self::BaseField>; 3]) -> [FieldElement<Self::BaseField>; 3] {
        assert!(
            !Self::is_neutral_element(p),
            "The point at infinity is not affine."
        );
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        [x / z, y / z, FieldElement::one()]
    }

    fn add(
        p: &[FieldElement<Self::BaseField>; 3],
        q: &[FieldElement<Self::BaseField>; 3],
    ) -> [FieldElement<Self::BaseField>; 3] {
        let (px, py, pz) = (&p[0], &p[1], &p[2]);
        let (qx, qy, qz) = (&q[0], &q[1], &q[2]);
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
                if u1 != u2 || *py == FieldElement::zero() {
                    Self::neutral_element()
                } else {
                    let w = Self::a() * pz.pow(2) + FieldElement::from(3) * px.pow(2);
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

    fn neg(p: &[FieldElement<Self::BaseField>; 3]) -> [FieldElement<Self::BaseField>; 3] {
        [p[0].clone(), -&p[1], p[2].clone()]
    }

    /// Evaluates the line between points `p` and `r` at point `q`
    fn line(
        p: &[FieldElement<Self::BaseField>; 3],
        r: &[FieldElement<Self::BaseField>; 3],
        q: &[FieldElement<Self::BaseField>; 3],
    ) -> FieldElement<Self::BaseField> {
        assert!(
            !Self::is_neutral_element(q),
            "q cannot be the point at infinity."
        );
        let (px, py, _) = (&p[0], &p[1], &p[2]);
        let (qx, qy, _) = (&q[0], &q[1], &q[2]);
        let (rx, ry, _) = (&r[0], &r[1], &r[2]);

        if Self::is_neutral_element(p) || Self::is_neutral_element(r) {
            if Self::eq(p, r) {
                return FieldElement::one();
            }
            if Self::is_neutral_element(p) {
                qx - rx
            } else {
                qx - px
            }
        } else if !Self::eq(p, r) {
            if px == rx {
                return qx - px;
            } else {
                let l = (ry - py) / (rx - px);
                return qy - py - l * (qx - px);
            }
        } else {
            let numerator = FieldElement::from(3) * &px.pow(2) + Self::a();
            let denominator = FieldElement::from(2) * py;
            if denominator == FieldElement::zero() {
                return qx - px;
            } else {
                let l = numerator / denominator;
                return qy - py - l * (qx - px);
            }
        }
    }

    /// Computes Miller's algorithm between points `p` and `q`.
    /// The implementaiton is based on Sagemath's sourcecode:
    /// See `_miller_` method on page 114
    /// https://www.sagemath.org/files/thesis/hansen-thesis-2009.pdf
    /// Other resources can be found at "Pairings for beginners" from Craig Costello, Algorithm 5.1, page 79.
    fn miller(
        p: &[FieldElement<Self::BaseField>; 3],
        q: &[FieldElement<Self::BaseField>; 3],
    ) -> FieldElement<Self::BaseField> {
        let p = Self::affine(p);
        let q = Self::affine(q);
        let mut order_r = Self::order_r();
        let mut bs = vec![];
        while order_r > 0 {
            bs.insert(0, order_r & 1);
            order_r >>= 1;
        }

        let mut f = FieldElement::one();
        let mut r = p.clone();

        for b in bs[1..].iter() {
            let s = Self::affine(&Self::add(&r, &r));
            f = f.pow(2) * (Self::line(&r, &r, &q) / Self::line(&s, &Self::neg(&s), &q));
            r = s;

            if *b == 1 {
                let mut s = Self::add(&r, &p);
                if s != Self::neutral_element() {
                    s = Self::affine(&s);
                }
                f = f * (Self::line(&r, &p, &q) / Self::line(&s, &Self::neg(&s), &q));
                r = s;
            }
        }
        f
    }

    /// Computes the Weil pairing between points `p` and `q`.
    /// See "Pairing for beginners" from Craig Costello, page 79.
    #[allow(unused)]
    fn weil_pairing(
        p: &[FieldElement<Self::BaseField>; 3],
        q: &[FieldElement<Self::BaseField>; 3],
    ) -> FieldElement<Self::BaseField> {
        if Self::is_neutral_element(p) || Self::is_neutral_element(q) || Self::eq(p, q) {
            FieldElement::one()
        } else {
            let numerator = Self::miller(p, q);
            let denominator = Self::miller(q, p);
            let result = numerator / denominator;
            -result
        }
    }

    /// Computes the Tate pairing between points `p` and `q`.
    /// See "Pairing for beginners" from Craig Costello, page 79.
    fn tate_pairing(
        p: &[FieldElement<Self::BaseField>; 3],
        q: &[FieldElement<Self::BaseField>; 3],
    ) -> FieldElement<Self::BaseField> {
        if Self::is_neutral_element(p) || Self::is_neutral_element(q) || Self::eq(p, q) {
            FieldElement::one()
        } else {
            Self::miller(p, q).pow(Self::target_normalization_power() as u128)
        }
    }
}

pub trait HasDistortionMap: HasEllipticCurveOperations {
    fn distorsion_map(p: &[FieldElement<Self::BaseField>; 3])
        -> [FieldElement<Self::BaseField>; 3];
}

/// Represents an elliptic curve point using the projective short Weierstrass form:
///   y^2 * z = x^3 + a * x * z^2 + b * z^3
/// x, y and z variables are field extension elements.
#[derive(Debug, Clone)]
pub struct EllipticCurveElement<E: HasEllipticCurveOperations> {
    value: [FieldElement<E::BaseField>; 3],
    elliptic_curve: PhantomData<E>,
}

impl<E: HasEllipticCurveOperations> EllipticCurveElement<E> {
    /// Creates an elliptic curve point giving the (x, y, z) coordinates.
    fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
        assert_eq!(
            E::defining_equation(&value),
            FieldElement::zero(),
            "Point ({:?}) does not belong to the elliptic curve.",
            &value
        );
        Self {
            value,
            elliptic_curve: PhantomData,
        }
    }

    #[allow(unused)]
    fn x(&self) -> &FieldElement<E::BaseField> {
        &self.value[0]
    }

    #[allow(unused)]
    fn y(&self) -> &FieldElement<E::BaseField> {
        &self.value[1]
    }

    #[allow(unused)]
    fn z(&self) -> &FieldElement<E::BaseField> {
        &self.value[2]
    }

    #[allow(unused)]
    fn to_affine(&self) -> Self {
        Self {
            value: E::affine(&self.value),
            elliptic_curve: PhantomData,
        }
    }

    #[allow(unused)]
    fn weil_pairing(&self, other: &Self) -> FieldElement<E::BaseField> {
        E::weil_pairing(&self.value, &other.value)
    }

    fn tate_pairing(&self, other: &Self) -> FieldElement<E::BaseField> {
        E::tate_pairing(&self.value, &other.value)
    }
}

impl<E: HasEllipticCurveOperations> PartialEq for EllipticCurveElement<E> {
    fn eq(&self, other: &Self) -> bool {
        E::eq(&self.value, &other.value)
    }
}

impl<E: HasEllipticCurveOperations> Eq for EllipticCurveElement<E> {}

impl<E: HasEllipticCurveOperations + HasDistortionMap> CyclicBilinearGroup
    for EllipticCurveElement<E>
{
    type PairingOutput = FieldElement<E::BaseField>;

    fn generator() -> Self {
        Self::new([
            E::generator_affine_x(),
            E::generator_affine_y(),
            FieldElement::one(),
        ])
    }

    fn neutral_element() -> Self {
        Self::new(E::neutral_element())
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from Moonmath (Algorithm 7, page 89)
    fn operate_with(&self, other: &Self) -> Self {
        Self::new(E::add(&self.value, &other.value))
    }

    /// Computes a Type 1 Tate pairing between `self` and `other.
    /// See "Pairing for beginners" from Craig Costello, section 4.2 Pairing types, page 58.
    /// Note that a distorsion map is applied to `other` before using the Tate pairing.
    /// So this method can be called with two field extension elements from the base field.
    fn pairing(&self, other: &Self) -> Self::PairingOutput {
        let [qx, qy, qz] = E::distorsion_map(&other.value);
        Self::tate_pairing(self, &Self::new([qx, qy, qz]))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        config::{ORDER_P, ORDER_R},
        field::{
            quadratic_extension::{
                HasQuadraticNonResidue, QuadraticExtensionField, QuadraticExtensionFieldElement,
            },
            u64_prime_field::{U64FieldElement, U64PrimeField},
        },
    };

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
    impl HasEllipticCurveOperations for CurrentCurve {
        type BaseField = QuadraticExtensionField<U64PrimeField<ORDER_P>, QuadraticNonResidue>;

        fn a() -> FieldElement<Self::BaseField> {
            FieldElement::from(1)
        }

        fn b() -> FieldElement<Self::BaseField> {
            FieldElement::from(0)
        }

        fn generator_affine_x() -> FieldElement<Self::BaseField> {
            FieldElement::from(35)
        }

        fn generator_affine_y() -> FieldElement<Self::BaseField> {
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

    impl HasDistortionMap for CurrentCurve {
        fn distorsion_map(
            p: &[FieldElement<Self::BaseField>; 3],
        ) -> [FieldElement<Self::BaseField>; 3] {
            let (x, y, z) = (&p[0], &p[1], &p[2]);
            let t = FieldElement::new([FieldElement::zero(), FieldElement::one()]);
            [-x, y * t, z.clone()]
        }
    }

    // This tests only apply for the specific curve found in the configuration file.
    #[test]
    fn create_valid_point_works() {
        let point =
            EllipticCurveElement::<CurrentCurve>::new([FEE::from(35), FEE::from(31), FEE::from(1)]);
        assert_eq!(*point.x(), FEE::new_base(35));
        assert_eq!(*point.y(), FEE::new_base(31));
        assert_eq!(*point.z(), FEE::new_base(1));
    }

    #[test]
    #[should_panic]
    fn create_invalid_points_panicks() {
        EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(0),
            FEE::new_base(1),
            FEE::new_base(1),
        ]);
    }

    #[test]
    fn equality_works() {
        let g = EllipticCurveElement::<CurrentCurve>::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
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
        assert_eq!(
            point_1,
            EllipticCurveElement::<CurrentCurve>::neutral_element()
        );
    }

    #[test]
    fn doubling_a_point_works() {
        let point = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(35),
            FEE::new_base(31),
            FEE::new_base(1),
        ]);
        let expected_result = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(25),
            FEE::new_base(29),
            FEE::new_base(1),
        ]);
        assert_eq!(point.operate_with_self(2).to_affine(), expected_result);
    }

    #[test]
    fn test_weil_pairing() {
        type FE = U64FieldElement<ORDER_P>;
        let pa = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(35),
            FEE::new_base(31),
            FEE::new_base(1),
        ]);
        let pb = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
            FEE::new_base(1),
        ]);
        let expected_result = FEE::new([FE::new(46), FE::new(3)]);

        let result_weil = EllipticCurveElement::<CurrentCurve>::weil_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }

    #[test]
    fn test_tate_pairing() {
        type FE = U64FieldElement<ORDER_P>;
        let pa = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(35),
            FEE::new_base(31),
            FEE::new_base(1),
        ]);
        let pb = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
            FEE::new_base(1),
        ]);
        let expected_result = FEE::new([FE::new(42), FE::new(19)]);

        let result_weil = EllipticCurveElement::<CurrentCurve>::tate_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }
}
