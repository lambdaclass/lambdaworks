use crate::cyclic_group::IsGroup;
use crate::{
    elliptic_curve::short_weierstrass::{
        point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass,
    },
    field::element::FieldElement,
    unsigned_integer::traits::IsUnsignedInteger,
};

/// Evaluates the Self::line between points `p` and `r` at point `q`
pub fn line<E: IsShortWeierstrass>(
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
pub fn miller<E: IsShortWeierstrass, I: IsUnsignedInteger>(
    order_r: &I,
    p: &ShortWeierstrassProjectivePoint<E>,
    q: &ShortWeierstrassProjectivePoint<E>,
) -> FieldElement<E::BaseField> {
    let mut order_r = *order_r;
    let p = p.to_affine();
    let q = q.to_affine();
    let mut bs = vec![];

    // TODO: this function compares with UnsignedInt that contain zeros and ones.
    // This unsigned ints might be 384 bit long or more. An idea to optimize this is
    // implementing a method for UnsignedInt that iterates over its bit representation.
    while order_r > I::from(0) {
        bs.insert(0, order_r & I::from(1));
        order_r = order_r >> 1;
    }

    let mut f = FieldElement::one();
    let mut r = p.clone();

    for b in bs[1..].iter() {
        let s = (r.operate_with(&r)).to_affine();
        f = f.pow(2_u16) * (line(&r, &r, &q) / line(&s, &s.neg(), &q));
        r = s;

        if b == &I::from(1) {
            let mut s = r.operate_with(&p);
            if !s.is_neutral_element() {
                s = s.to_affine();
            }
            f = f * (line(&r, &p, &q) / line(&s, &s.neg(), &q));
            r = s;
        }
    }
    f
}
