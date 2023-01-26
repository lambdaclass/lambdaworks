use crate::field::element::FieldElement;
use crate::field::traits::HasFieldOperations;
use std::fmt::Debug;

/// Trait to add elliptic curves behaviour to a struct.
/// We use the short Weierstrass form equation: `y^2 = x^3 + a * x  + b`.
pub trait HasEllipticCurveOperations: Clone + Debug {
    type BaseField: HasFieldOperations + Clone + Debug;

    /// `a` coefficient for the equation `y^2 = x^3 + a * x  + b`.
    fn a() -> FieldElement<Self::BaseField>;

    /// `b` coefficient for the equation  `y^2 = x^3 + a * x  + b`.
    fn b() -> FieldElement<Self::BaseField>;

    /// `x` component of the generator (x, y) in affine form.
    fn generator_affine_x() -> FieldElement<Self::BaseField>;

    /// `y` component of the generator (x, y) in affine form.
    fn generator_affine_y() -> FieldElement<Self::BaseField>;

    /// The embedding degree of the curve. This is the minimum integer `k`
    /// for which `order_r` divides `order_p.pow(k) - 1`.
    fn embedding_degree() -> u32;

    /// Order of the subgroup of the curve (e.g.: number of elements in
    /// the subgroup of the curve).
    fn order_r() -> u64;

    /// Order of the base field (e.g.: order of the field where `a` and `b` are defined).
    fn order_p() -> u64;

    /// Order of the field extension where the entire `order_r` torsion
    /// group lives.
    fn order_field_extension() -> u64 {
        Self::order_p().pow(Self::embedding_degree())
    }

    /// Normalization power for the Tate pairing.
    fn target_normalization_power() -> u64 {
        (Self::order_field_extension() - 1) / Self::order_r()
    }

    /// Evaluates the short Weierstrass equation at (x, y z).
    /// Used for checking if [x: y: z] belongs to the elliptic curve.
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

    /// Returns the additive inverse of the projective point `p`
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

/// Trait to add distortion maps to Elliptic Curves.
/// Typically used to support type I pairings.
pub trait HasDistortionMap: HasEllipticCurveOperations {
    fn distorsion_map(p: &[FieldElement<Self::BaseField>; 3])
        -> [FieldElement<Self::BaseField>; 3];
}
