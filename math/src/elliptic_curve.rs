use crate::{config::{
    ELLIPTIC_CURVE_A, ELLIPTIC_CURVE_B, GENERATOR_AFFINE_X, GENERATOR_AFFINE_Y, ORDER_P, ORDER_R,
    TARGET_NORMALIZATION_POWER,
}, field_extension_element::{QuadraticNonResidue, QuadraticExtensionField}, algebraic_element::FieldElement, field_element::U64PrimeField};

use super::{
    cyclic_group::CyclicBilinearGroup, field_element::U64FieldElement,
};
use std::ops;

#[derive(Debug, Clone)]
pub struct MyQuadraticNonResidue;
impl QuadraticNonResidue<U64PrimeField<ORDER_P>> for MyQuadraticNonResidue {
    fn quadratic_non_residue() -> FieldElement<U64PrimeField<ORDER_P>> {
        -FieldElement::one()
    }
}

type FE = U64FieldElement<ORDER_P>;
type MyFieldExtension = QuadraticExtensionField<U64PrimeField<ORDER_P>, MyQuadraticNonResidue>;
#[allow(clippy::upper_case_acronyms)]
type FEE =FieldElement<MyFieldExtension>;

/// Represents an elliptic curve point using the projective short Weierstrass form:
///   y^2 * z = x^3 + a * x * z^2 + b * z^3
/// x, y and z variables are field extension elements.
#[derive(Debug, Clone)]
pub struct EllipticCurveElement {
    x: FEE,
    y: FEE,
    z: FEE,
}

impl EllipticCurveElement {
    /// Creates an elliptic curve point giving the (x, y, z) coordinates.
    fn new(x: FEE, y: FEE, z: FEE) -> Self {
        assert_eq!(
            Self::defining_equation(&x, &y, &z),
            FEE::zero(),
            "Point ({:?}, {:?}, {:?}) does not belong to the elliptic curve.",
            x,
            y,
            z
        );
        Self { x, y, z }
    }

    /// Evaluates the short Weierstrass equation at (x, y z).
    /// Useful for checking if (x, y, z) belongs to the elliptic curve.
    fn defining_equation(x: &FEE, y: &FEE, z: &FEE) -> FEE {
        y.pow(2) * z
            - x.pow(3)
            - FEE::new_base(&FE::new(ELLIPTIC_CURVE_A)) * x * z.pow(2)
            - FEE::new_base(&FE::new(ELLIPTIC_CURVE_B)) * z.pow(3)
    }

    /// Normalize the projective coordinates to obtain affine coordinates
    /// of the form (x, y, 1)
    /// Panics if `self` is the point at infinity
    fn affine(&self) -> Self {
        assert!(
            self != &EllipticCurveElement::neutral_element(),
            "The point at infinity is not affine."
        );
        Self::new(&self.x / &self.z, &self.y / &self.z, FEE::new_base(&FE::new(1)))
    }

    /// Evaluates the line between points `self` and `r` at point `q`
    fn line(&self, r: &Self, q: &Self) -> FEE {
        assert_ne!(
            *q,
            Self::neutral_element(),
            "q cannot be the point at infinity."
        );
        if *self == Self::neutral_element() || *r == Self::neutral_element() {
            if self == r {
                return FEE::new_base(&FE::new(1));
            }
            if *self == Self::neutral_element() {
                &q.x - &r.x
            } else {
                &q.x - &self.x
            }
        } else if self != r {
            if self.x == r.x {
                return &q.x - &self.x;
            } else {
                let l = (&r.y - &self.y) / (&r.x - &self.x);
                return &q.y - &self.y - l * (&q.x - &self.x);
            }
        } else {
            let numerator = FEE::new_base(&FE::new(3)) * &self.x.pow(2) + FEE::new_base(&FE::new(ELLIPTIC_CURVE_A));
            let denominator = FEE::new_base(&FE::new(2)) * &self.y;
            if denominator == FEE::new_base(&FE::new(0)) {
                return &q.x - &self.x;
            } else {
                let l = numerator / denominator;
                return &q.y - &self.y - l * (&q.x - &self.x);
            }
        }
    }

    /// Computes Miller's algorithm between points `p` and `q`.
    /// The implementaiton is based on Sagemath's sourcecode:
    /// See `_miller_` method on page 114
    /// https://www.sagemath.org/files/thesis/hansen-thesis-2009.pdf
    /// Other resources can be found at "Pairings for beginners" from Craig Costello, Algorithm 5.1, page 79.
    fn miller(p: &Self, q: &Self) -> FEE {
        let p = p.affine();
        let q = q.affine();
        let mut order_r = ORDER_R;
        let mut bs = vec![];
        while order_r > 0 {
            bs.insert(0, order_r & 1);
            order_r >>= 1;
        }

        let mut f = FEE::new_base(&FE::new(1));
        let mut r = p.clone();

        for b in bs[1..].iter() {
            let s = r.operate_with(&r).affine();
            f = f.pow(2) * (r.line(&r, &q) / s.line(&-(&s), &q));
            r = s;

            if *b == 1 {
                let mut s = r.operate_with(&p);
                if s != Self::neutral_element() {
                    s = s.affine();
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
    fn weil_pairing(p: &Self, q: &Self) -> FEE {
        if *p == Self::neutral_element() || *q == Self::neutral_element() || p == q {
            FEE::new_base(&FE::new(1))
        } else {
            let numerator = Self::miller(p, q);
            let denominator = Self::miller(q, p);
            let result = numerator / denominator;
            -result
        }
    }

    /// Computes the Tate pairing between points `p` and `q`.
    /// See "Pairing for beginners" from Craig Costello, page 79.
    fn tate_pairing(p: &Self, q: &Self) -> FEE {
        if *p == Self::neutral_element() || *q == Self::neutral_element() || p == q {
            FEE::new_base(&FE::new(1))
        } else {
            Self::miller(p, q).pow(TARGET_NORMALIZATION_POWER as u128)
        }
    }

    /// Apply a distorsion map to point `p`.
    /// This is useful for converting points living in the base field
    /// to points living in the extension field.
    /// The current implementation only works for the elliptic curve with A=1 and B=0
    /// ORDER_P=59. This curve was chosen because it is supersingular.
    fn distorsion_map(p: &Self) -> Self {
        let t = FEE::new([FieldElement::zero(), FieldElement::one()]);
        Self::new(-&p.x, &p.y * t, p.z.clone())
    }
}

impl PartialEq for EllipticCurveElement {
    fn eq(&self, other: &Self) -> bool {
        // Projective equality relation: first point has to be a multiple of the other
        (&self.x * &other.z == &self.z * &other.x) && (&self.x * &other.y == &self.y * &other.x)
    }
}
impl Eq for EllipticCurveElement {}

impl ops::Neg for &EllipticCurveElement {
    type Output = EllipticCurveElement;

    fn neg(self) -> Self::Output {
        Self::Output::new(self.x.clone(), -self.y.clone(), self.z.clone())
    }
}

impl ops::Neg for EllipticCurveElement {
    type Output = EllipticCurveElement;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl CyclicBilinearGroup for EllipticCurveElement {
    type PairingOutput = FEE;

    fn generator() -> Self {
        Self::new(
            FEE::new_base(&FE::new(GENERATOR_AFFINE_X)),
            FEE::new_base(&FE::new(GENERATOR_AFFINE_Y)),
            FEE::new_base(&FE::new(1)),
        )
    }

    fn neutral_element() -> Self {
        Self::new(FEE::new_base(&FE::new(0)), FEE::new_base(&FE::new(1)), FEE::new_base(&FE::new(0)))
    }

    fn operate_with_self(&self, times: u128) -> Self {
        let mut times = times;
        let mut result = Self::neutral_element();
        let mut base = self.clone();

        while times > 0 {
            // times % 2 == 1
            if times & 1 == 1 {
                result = result.operate_with(&base);
            }
            // times = times / 2
            times >>= 1;
            base = base.operate_with(&base);
        }
        result
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from Moonmath (Algorithm 7, page 89)
    fn operate_with(&self, other: &Self) -> Self {
        if *other == Self::neutral_element() {
            self.clone()
        } else if *self == Self::neutral_element() {
            other.clone()
        } else {
            let u1 = &other.y * &self.z;
            let u2 = &self.y * &other.z;
            let v1 = &other.x * &self.z;
            let v2 = &self.x * &other.z;
            if v1 == v2 {
                if u1 != u2 || self.y == FEE::new_base(&FE::new(0)) {
                    Self::neutral_element()
                } else {
                    let w = FEE::new_base(&FE::new(ELLIPTIC_CURVE_A)) * self.z.pow(2)
                        + FEE::new_base(&FE::new(3)) * self.x.pow(2);
                    let s = &self.y * &self.z;
                    let b = &self.x * &self.y * &s;
                    let h = w.pow(2) - FEE::new_base(&FE::new(8)) * &b;
                    let xp = FEE::new_base(&FE::new(2)) * &h * &s;
                    let yp = w * (FEE::new_base(&FE::new(4)) * &b - &h)
                        - FEE::new_base(&FE::new(8)) * self.y.pow(2) * s.pow(2);
                    let zp = FEE::new_base(&FE::new(8)) * s.pow(3);
                    Self::new(xp, yp, zp)
                }
            } else {
                let u = u1 - &u2;
                let v = v1 - &v2;
                let w = &self.z * &other.z;
                let a = u.pow(2) * &w - v.pow(3) - FEE::new_base(&FE::new(2)) * v.pow(2) * &v2;
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
        Self::tate_pairing(self, &Self::distorsion_map(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // This tests only apply for the specific curve found in the configuration file.
    #[test]
    fn create_valid_point_works() {
        let point =
            EllipticCurveElement::new(FEE::new_base(&FE::new(35)), FEE::new_base(&FE::new(31)), FEE::new_base(&FE::new(1)));
        assert_eq!(point.x, FEE::new_base(&FE::new(35)));
        assert_eq!(point.y, FEE::new_base(&FE::new(31)));
        assert_eq!(point.z, FEE::new_base(&FE::new(1)));
    }

    #[test]
    #[should_panic]
    fn create_invalid_points_panicks() {
        EllipticCurveElement::new(FEE::new_base(&FE::new(0)), FEE::new_base(&FE::new(1)), FEE::new_base(&FE::new(1)));
    }

    #[test]
    fn operate_with_self_works() {
        let mut point_1 = EllipticCurveElement::generator();
        point_1 = point_1.operate_with_self(ORDER_R as u128);
        assert_eq!(point_1, EllipticCurveElement::neutral_element());
    }

    #[test]
    fn doubling_a_point_works() {
        let point =
            EllipticCurveElement::new(FEE::new_base(&FE::new(35)), FEE::new_base(&FE::new(31)), FEE::new_base(&FE::new(1)));
        let expected_result =
            EllipticCurveElement::new(FEE::new_base(&FE::new(25)), FEE::new_base(&FE::new(29)), FEE::new_base(&FE::new(1)));
        assert_eq!(point.operate_with_self(2).affine(), expected_result);
    }

    #[test]
    fn test_weil_pairing() {
        let pa = EllipticCurveElement::new(FEE::new_base(&FE::new(35)), FEE::new_base(&FE::new(31)), FEE::new_base(&FE::new(1)));
        let pb = EllipticCurveElement::new(
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
            FEE::new_base(&FE::new(1)),
        );
        let expected_result = FEE::new([FE::new(46), FE::new(3)]);

        let result_weil = EllipticCurveElement::weil_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }

    #[test]
    fn test_tate_pairing() {
        let pa = EllipticCurveElement::new(FEE::new_base(&FE::new(35)), FEE::new_base(&FE::new(31)), FEE::new_base(&FE::new(1)));
        let pb = EllipticCurveElement::new(
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
            FEE::new_base(&FE::new(1)),
        );
        let expected_result = FEE::new([FE::new(42), FE::new(19)]);

        let result_weil = EllipticCurveElement::tate_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }
}
