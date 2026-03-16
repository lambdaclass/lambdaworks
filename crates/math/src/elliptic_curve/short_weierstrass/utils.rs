use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::{
    ShortWeierstrassJacobianPoint, ShortWeierstrassProjectivePoint,
};
use crate::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
use crate::field::element::FieldElement;
use crate::unsigned_integer::element::U256;

/// Precomputed constants for Babai-rounding GLV scalar decomposition.
///
/// Given the GLV lattice L = {(a,b) : a + b·λ ≡ 0 (mod r)} with short basis vectors
/// v1 = (v1_0, v1_1) and v2 = (v2_0, v2_1) found via LLL, the Babai nearest-plane
/// algorithm decomposes any scalar k into k = k1 + k2·λ (mod r) with |k1|, |k2| ~ √r.
///
/// Rounding constants: q1 = round(2^256 · v2_1 / r), q2 = round(2^256 · (−v1_1) / r).
/// We store absolute values and sign flags for all components.
///
/// References:
/// - GLV method: Gallant, Lambert, Vanstone. "Faster Point Multiplication on Elliptic
///   Curves with Efficient Endomorphisms". CRYPTO 2001. <https://doi.org/10.1007/3-540-44647-8_11>
/// - GLS extension: Galbraith, Lin, Scott. "Endomorphisms for Faster Elliptic Curve
///   Cryptography on a Large Class of Curves". EUROCRYPT 2009. <https://eprint.iacr.org/2008/194>
/// - Babai rounding: see Constantine `constantine/math/endomorphisms/split_scalars.nim`
///   <https://github.com/mratsim/constantine/blob/master/constantine/math/endomorphisms/split_scalars.nim>
pub(crate) struct GlvDecompConstants {
    /// Babai rounding constant |q1| = |round(2^256 · v2[1] / r)|
    pub q1: U256,
    /// Babai rounding constant |q2| = |round(2^256 · (−v1[1]) / r)|
    pub q2: U256,
    /// |v1[0]|
    pub b1_0: U256,
    /// |v1[1]|
    pub b1_1: U256,
    /// |v2[0]|
    pub b2_0: U256,
    /// |v2[1]|
    pub b2_1: U256,
    /// true if v1[0] < 0
    pub v1_0_is_neg: bool,
    /// true if v1[1] < 0
    pub v1_1_is_neg: bool,
    /// true if v2[0] < 0
    pub v2_0_is_neg: bool,
    /// true if v2[1] < 0
    pub v2_1_is_neg: bool,
    /// true if q1 < 0
    pub q1_is_neg: bool,
    /// true if q2 < 0
    pub q2_is_neg: bool,
}

/// Signed addition: computes (acc_neg, acc) += (term_neg, term) using unsigned arithmetic.
/// Returns the new (is_negative, magnitude).
#[inline(always)]
fn signed_add(acc_neg: bool, acc: U256, term_neg: bool, term: U256) -> (bool, U256) {
    if acc_neg == term_neg {
        // Same sign: add magnitudes, keep sign
        (acc_neg, U256::add(&acc, &term).0)
    } else {
        // Different signs: subtract smaller from larger
        if acc >= term {
            (acc_neg, U256::sub(&acc, &term).0)
        } else {
            (term_neg, U256::sub(&term, &acc).0)
        }
    }
}

/// Babai nearest-plane GLV scalar decomposition.
///
/// Decomposes k into k1 + k2·λ (mod r) where |k1|, |k2| ≈ √r (~128 bits for 256-bit r).
///
/// Returns `(k1_neg, |k1|, k2_neg, |k2|)`.
///
/// # Algorithm
///
/// 1. c1 = ⌊k · |q1| / 2^256⌋ (approximate rounding via widening multiply high word)
/// 2. c2 = ⌊k · |q2| / 2^256⌋
/// 3. k1 = k − (sign(q1)·c1)·v1[0] − (sign(q2)·c2)·v2[0]
/// 4. k2 = −(sign(q1)·c1)·v1[1] − (sign(q2)·c2)·v2[1]
///
/// References: kilic/bls12-381, gnark-crypto, Constantine
/// (`constantine/math/endomorphisms/split_scalars.nim`).
pub(crate) fn glv_decompose_babai(k: &U256, c: &GlvDecompConstants) -> (bool, U256, bool, U256) {
    // c1 = hi(k * |q1|), sign inherited from q1
    let (c1, _) = U256::mul(k, &c.q1);
    // c2 = hi(k * |q2|), sign inherited from q2
    let (c2, _) = U256::mul(k, &c.q2);

    // Compute products: c_i * |basis_component|, then assign signs.
    // c1 has sign of q1, c2 has sign of q2.

    // term1 = c1 * v1[0]: sign = q1_sign XOR v1_0_sign (XOR because neg*neg=pos, etc.)
    //   but we want magnitude only here
    let c1_b10 = U256::mul(&c1, &c.b1_0).1; // low word (hi should be 0 for well-formed constants)
    let c1_b10_neg = c.q1_is_neg ^ c.v1_0_is_neg;

    let c2_b20 = U256::mul(&c2, &c.b2_0).1;
    let c2_b20_neg = c.q2_is_neg ^ c.v2_0_is_neg;

    // k1 = k - c1*v1[0] - c2*v2[0]
    // Start: acc = (+, k)
    // Subtract term1: acc = acc + (-term1_sign, term1_mag)
    let (k1_neg, k1_val) = signed_add(false, *k, !c1_b10_neg, c1_b10);
    let (k1_neg, k1_val) = signed_add(k1_neg, k1_val, !c2_b20_neg, c2_b20);

    // k2 = -c1*v1[1] - c2*v2[1]
    // -c1*v1[1]: sign = !(q1_sign XOR v1_1_sign) because of the leading minus
    let c1_b11 = U256::mul(&c1, &c.b1_1).1;
    let c1_b11_neg = !(c.q1_is_neg ^ c.v1_1_is_neg);

    let c2_b21 = U256::mul(&c2, &c.b2_1).1;
    let c2_b21_neg = !(c.q2_is_neg ^ c.v2_1_is_neg);

    // Start: acc = (false, 0), then add both terms
    let (k2_neg, k2_val) = signed_add(c1_b11_neg, c1_b11, c2_b21_neg, c2_b21);

    (k1_neg, k1_val, k2_neg, k2_val)
}

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
/// Cost: 2M + 1S.
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
/// Cost: 2M + 1S.
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
