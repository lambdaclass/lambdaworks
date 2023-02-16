use crate::{
    cyclic_group::IsGroup, field::element::FieldElement,
    unsigned_integer::traits::IsUnsignedInteger,
};

use super::{element::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass};

pub trait HasTypeIPairing {
    type UIntOrders: IsUnsignedInteger;
    fn order_r() -> Self::UIntOrders;
    fn target_normalization_power() -> Vec<u64>;

    /// Evaluates the Self::line between points `p` and `r` at point `q`
    fn line<E: IsShortWeierstrass>(
        p: &ShortWeierstrassProjectivePoint<E>,
        r: &ShortWeierstrassProjectivePoint<E>,
        q: &ShortWeierstrassProjectivePoint<E>,
    ) -> FieldElement<E::BaseField> {
        // TODO: Improve error handling.
        debug_assert!(
            !q.is_neutral_element(),
            "q cannot be the point at infinity."
        );
        let [px, py, _] = p.coordinates();
        let [qx, qy, _] = q.coordinates();
        let [rx, ry, _] = r.coordinates();

        if p.is_neutral_element() || r.is_neutral_element() {
            if p == r {
                return FieldElement::one();
            }
            if p.is_neutral_element() {
                qx - rx
            } else {
                qx - px
            }
        } else if p != r {
            if px == rx {
                qx - px
            } else {
                let l = (ry - py) / (rx - px);
                qy - py - l * (qx - px)
            }
        } else {
            let numerator = FieldElement::from(3) * &px.pow(2_u16) + E::a();
            let denominator = FieldElement::from(2) * py;
            if denominator == FieldElement::zero() {
                qx - px
            } else {
                let l = numerator / denominator;
                qy - py - l * (qx - px)
            }
        }
    }

    /// Computes Miller's algorithm between points `p` and `q`.
    /// The implementaiton is based on Sagemath's sourcecode:
    /// See `_miller_` method on page 114
    /// https://www.sagemath.org/files/thesis/hansen-thesis-2009.pdf
    /// Other resources can be found at "Pairings for beginners" from Craig Costello, Algorithm 5.1, page 79.
    fn miller<E: IsShortWeierstrass>(
        p: &ShortWeierstrassProjectivePoint<E>,
        q: &ShortWeierstrassProjectivePoint<E>,
    ) -> FieldElement<E::BaseField> {
        let p = p.to_affine();
        let q = q.to_affine();
        let mut order_r = Self::order_r();
        let mut bs = vec![];

        // TODO: this function compares with UnsignedInt that contain zeros and ones.
        // This unsigned ints might be 384 bit long or more. An idea to optimize this is
        // implementing a method for UnsignedInt that iterates over its bit representation.
        while order_r > Self::UIntOrders::from(0) {
            bs.insert(0, order_r & Self::UIntOrders::from(1));
            order_r = order_r >> 1;
        }

        let mut f = FieldElement::one();
        let mut r = p.clone();

        for b in bs[1..].iter() {
            let s = (r.operate_with(&r)).to_affine();
            f = f.pow(2_u16) * (Self::line(&r, &r, &q) / Self::line(&s, &s.neg(), &q));
            r = s;

            if *b == Self::UIntOrders::from(1) {
                let mut s = r.operate_with(&p);
                if !s.is_neutral_element() {
                    s = s.to_affine();
                }
                f = f * (Self::line(&r, &p, &q) / Self::line(&s, &s.neg(), &q));
                r = s;
            }
        }
        f
    }

    /// Computes the Weil pairing between points `p` and `q`.
    /// See "Pairing for beginners" from Craig Costello, page 79.
    #[allow(unused)]
    fn weil_pairing<E: IsShortWeierstrass>(
        p: &ShortWeierstrassProjectivePoint<E>,
        q: &ShortWeierstrassProjectivePoint<E>,
    ) -> FieldElement<E::BaseField> {
        if p.is_neutral_element() || q.is_neutral_element() || p == q {
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
    fn tate_pairing<E: IsShortWeierstrass>(
        p: &ShortWeierstrassProjectivePoint<E>,
        q: &ShortWeierstrassProjectivePoint<E>,
    ) -> FieldElement<E::BaseField> {
        if p.is_neutral_element() || q.is_neutral_element() || p == q {
            FieldElement::one()
        } else {
            let mut base = Self::miller(p, q);
            let bit_representation_exponent = Self::target_normalization_power();
            let mut pow = FieldElement::one();

            // This is computes the power of base raised to the target_normalization_power
            for (index, limb) in bit_representation_exponent.iter().rev().enumerate() {
                let mut limb = *limb;
                for _bit in 1..=16 {
                    if limb & 1 == 1 {
                        pow = &pow * &base;
                    }
                    base = &base * &base;
                    let finished = (index == bit_representation_exponent.len() - 1) && (limb == 0);
                    if !finished {
                        limb >>= 1;
                    } else {
                        break;
                    }
                }
            }
            pow
        }
    }
}
