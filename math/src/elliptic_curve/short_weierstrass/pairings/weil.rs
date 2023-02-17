use crate::cyclic_group::IsGroup;
use crate::{
    elliptic_curve::short_weierstrass::{
        pairings::miller::miller, point::ShortWeierstrassProjectivePoint,
        traits::IsShortWeierstrass,
    },
    field::element::FieldElement,
    unsigned_integer::traits::IsUnsignedInteger,
};

/// Computes the Weil pairing between points `p` and `q`.
/// See "Pairing for beginners" from Craig Costello, page 79.
#[allow(unused)]
pub fn weil_pairing<E: IsShortWeierstrass, I: IsUnsignedInteger>(
    order_r: &I,
    p: &ShortWeierstrassProjectivePoint<E>,
    q: &ShortWeierstrassProjectivePoint<E>,
) -> FieldElement<E::BaseField> {
    if p.is_neutral_element() || q.is_neutral_element() || p == q {
        FieldElement::one()
    } else {
        let numerator = miller(order_r, p, q);
        let denominator = miller(order_r, q, p);
        let result = numerator / denominator;
        -result
    }
}
