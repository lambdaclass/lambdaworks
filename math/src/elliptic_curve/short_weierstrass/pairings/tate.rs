use super::miller::miller;
use crate::cyclic_group::IsGroup;
use crate::{
    elliptic_curve::short_weierstrass::{
        point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass,
    },
    field::element::FieldElement,
    unsigned_integer::traits::IsUnsignedInteger,
};

/// Computes the Tate pairing between points `p` and `q`.
/// See "Pairing for beginners" from Craig Costello, page 79.
pub fn tate_pairing<E: IsShortWeierstrass, I: IsUnsignedInteger>(
    order_r: &I,
    final_exponentiation_power: Vec<u64>,
    p: &ShortWeierstrassProjectivePoint<E>,
    q: &ShortWeierstrassProjectivePoint<E>,
) -> FieldElement<E::BaseField> {
    if p.is_neutral_element() || q.is_neutral_element() || p == q {
        FieldElement::one()
    } else {
        let mut base = miller(order_r, p, q);
        let bit_representation_exponent = final_exponentiation_power;
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
