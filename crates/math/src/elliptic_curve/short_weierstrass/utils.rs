use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::{
    ShortWeierstrassJacobianPoint, ShortWeierstrassProjectivePoint,
};
use crate::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
use crate::field::element::FieldElement;
use crate::unsigned_integer::element::U256;

/// Gets bit at position `pos` from a U256 (little-endian bit indexing).
#[inline(always)]
pub(crate) fn get_bit(n: &U256, pos: usize) -> bool {
    if pos >= 256 {
        return false;
    }
    let limb_idx = 3 - pos / 64;
    let bit_idx = pos % 64;
    (n.limbs[limb_idx] >> bit_idx) & 1 == 1
}

/// Shamir's trick: computes [k1]P1 + [k2]P2 using joint double-and-add.
/// Uses Jacobian coordinates for efficient doubling (2M+5S vs 7M+5S in projective).
pub(crate) fn shamir_two_scalar_mul<C: IsShortWeierstrass>(
    p1: &ShortWeierstrassJacobianPoint<C>,
    k1: &U256,
    p2: &ShortWeierstrassJacobianPoint<C>,
    k2: &U256,
) -> ShortWeierstrassJacobianPoint<C> {
    let p1_plus_p2 = p1.operate_with(p2);
    let max_len = core::cmp::max(k1.bits_le(), k2.bits_le());

    if max_len == 0 {
        return ShortWeierstrassJacobianPoint::neutral_element();
    }

    let mut result = ShortWeierstrassJacobianPoint::neutral_element();

    for i in (0..max_len).rev() {
        result = result.double();

        match (get_bit(k1, i), get_bit(k2, i)) {
            (false, false) => {}
            (true, false) => result = result.operate_with(p1),
            (false, true) => result = result.operate_with(p2),
            (true, true) => result = result.operate_with(&p1_plus_p2),
        }
    }

    result
}

/// Converts a projective G1/G2 point to Jacobian for efficient doubling.
/// Projective (X:Y:Z) where affine=(X/Z, Y/Z) → Jacobian (X·Z : Y·Z² : Z).
/// Cost: 1M + 1S.
pub(crate) fn proj_to_jac<C: IsShortWeierstrass>(
    p: &ShortWeierstrassProjectivePoint<C>,
) -> ShortWeierstrassJacobianPoint<C> {
    let [x, y, z] = p.coordinates();
    if z == &FieldElement::zero() {
        return ShortWeierstrassJacobianPoint::neutral_element();
    }
    let z_sq = z.square();
    ShortWeierstrassJacobianPoint::new_unchecked([x * z, y * &z_sq, z.clone()])
}

/// Converts a Jacobian result back to projective.
/// Jacobian (X:Y:Z) where affine=(X/Z², Y/Z³) → Projective (X·Z : Y : Z³).
/// Cost: 1M + 1S.
pub(crate) fn jac_to_proj<C: IsShortWeierstrass>(
    p: ShortWeierstrassJacobianPoint<C>,
) -> ShortWeierstrassProjectivePoint<C> {
    let [x, y, z] = p.coordinates();
    if z == &FieldElement::zero() {
        return ShortWeierstrassProjectivePoint::neutral_element();
    }
    let z_sq = z.square();
    ShortWeierstrassProjectivePoint::new_unchecked([x * z, y.clone(), &z_sq * z])
}
