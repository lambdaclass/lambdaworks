use super::{
    curve::BLS12381Curve,
    field_extension::{Degree12ExtensionField, Degree2ExtensionField},
    twist::BLS12381TwistCurve,
};
use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass,
    elliptic_curve::short_weierstrass::{
        curves::bls12_381::field_extension::Degree6ExtensionField,
        point::ShortWeierstrassProjectivePoint,
    },
    field::element::FieldElement,
    unsigned_integer::element::UnsignedInteger,
};

/// This is equal to the frobenius trace of the BLS12 381 curve minus one.
const MILLER_LOOP_CONSTANT: u64 = 0xd201000000010000;

/// Implements the miller loop for the ate pairing of the BLS12 381 curve.
/// Based on algorithm 9.2, page 212 of the book
/// "Topics in computational number theory" by W. Bons and K. Lenstra
#[allow(unused)]
fn miller(
    q: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
) -> FieldElement<Degree12ExtensionField> {
    let mut r = q.clone();
    let mut f = FieldElement::<Degree12ExtensionField>::one();
    let mut miller_loop_constant = MILLER_LOOP_CONSTANT;
    let mut miller_loop_constant_bits: Vec<bool> = vec![];

    while miller_loop_constant > 0 {
        miller_loop_constant_bits.insert(0, (miller_loop_constant & 1) == 1);
        miller_loop_constant >>= 1;
    }

    for bit in miller_loop_constant_bits[1..].iter() {
        f = f.pow(2_u64) * line(&r, &r, p);
        r = r.operate_with(&r).to_affine();
        if *bit {
            f = f * line(&r, q, p);
            r = r.operate_with(q);
            if !r.is_neutral_element() {
                r = r.to_affine();
            }
        }
    }
    f.inv()
}

/// Auxiliary function for the final exponentiation of the ate pairing.
fn frobenius_square(
    f: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    let [a, b] = f.value();
    let w_raised_to_p_squared_minus_one = FieldElement::<Degree6ExtensionField>::new_base("1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaad");
    let omega_3 = FieldElement::<Degree2ExtensionField>::new_base("1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaac");
    let omega_3_squared = FieldElement::<Degree2ExtensionField>::new_base(
        "5f19672fdf76ce51ba69c6076a0f77eaddb3a93be6f89688de17d813620a00022e01fffffffefffe",
    );

    let [a0, a1, a2] = a.value();
    let [b0, b1, b2] = b.value();

    let f0 = FieldElement::new([a0.clone(), a1 * &omega_3, a2 * &omega_3_squared]);
    let f1 = FieldElement::new([b0.clone(), b1 * omega_3, b2 * omega_3_squared]);

    FieldElement::new([f0, f1 * w_raised_to_p_squared_minus_one])
}

// To understand more about how to reduce the final exponentiation
// read "Efficient Final Exponentiation via Cyclotomic Structure for
// Pairings over Families of Elliptic Curves" (https://eprint.iacr.org/2020/875.pdf)
//
// TODO: implement optimizations for the hard part of the final exponentiation.
#[allow(unused)]
fn final_exponentiation(
    base: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    const PHI_DIVIDED_BY_R: UnsignedInteger<20> = UnsignedInteger::from("f686b3d807d01c0bd38c3195c899ed3cde88eeb996ca394506632528d6a9a2f230063cf081517f68f7764c28b6f8ae5a72bce8d63cb9f827eca0ba621315b2076995003fc77a17988f8761bdc51dc2378b9039096d1b767f17fcbde783765915c97f36c6f18212ed0b283ed237db421d160aeb6a1e79983774940996754c8c71a2629b0dea236905ce937335d5b68fa9912aae208ccf1e516c3f438e3ba79");

    let f1 = base.conjugate() * base.inv();
    let f2 = frobenius_square(&f1) * f1;
    f2.pow(PHI_DIVIDED_BY_R)
}

/// Compute the ate pairing between point `p` in G1 and `q` in G2.
#[allow(unused)]
pub fn ate(
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    q: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
) -> FieldElement<Degree12ExtensionField> {
    batch_ate(&[(p, q)])
}

/// Compute the product of the ate pairings for a list of point pairs.
pub fn batch_ate(
    pairs: &[(
        &ShortWeierstrassProjectivePoint<BLS12381Curve>,
        &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    )],
) -> FieldElement<Degree12ExtensionField> {
    let mut result = FieldElement::one();
    for (p, q) in pairs {
        result = result * miller(q, p);
    }
    final_exponentiation(&result)
}

/// Evaluates the line between points `p` and `r` at point `q`
pub fn line(
    p: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    r: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    q: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
) -> FieldElement<Degree12ExtensionField> {
    // TODO: Improve error handling.
    debug_assert!(
        !q.is_neutral_element(),
        "q cannot be the point at infinity."
    );
    let [px, py] = p.to_fp12_unnormalized();
    let [rx, ry] = r.to_fp12_unnormalized();
    let [qx_fp, qy_fp, _] = q.coordinates().clone();
    let qx = FieldElement::<Degree12ExtensionField>::new([
        FieldElement::new([
            FieldElement::new([qx_fp, FieldElement::zero()]),
            FieldElement::zero(),
            FieldElement::zero(),
        ]),
        FieldElement::zero(),
    ]);
    let qy = FieldElement::<Degree12ExtensionField>::new([
        FieldElement::new([
            FieldElement::new([qy_fp, FieldElement::zero()]),
            FieldElement::zero(),
            FieldElement::zero(),
        ]),
        FieldElement::zero(),
    ]);
    let a_of_curve = FieldElement::<Degree12ExtensionField>::new([
        FieldElement::new([
            FieldElement::new([BLS12381Curve::a(), FieldElement::zero()]),
            FieldElement::zero(),
            FieldElement::zero(),
        ]),
        FieldElement::zero(),
    ]);

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
            let l = (ry - &py) / (rx - &px);
            qy - py - l * (qx - px)
        }
    } else {
        let numerator = FieldElement::from(3) * &px.pow(2_u16) + a_of_curve;
        let denominator = FieldElement::from(2) * &py;
        if denominator == FieldElement::zero() {
            qx - px
        } else {
            let l = numerator / denominator;
            qy - py - l * (qx - px)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{elliptic_curve::traits::IsEllipticCurve, unsigned_integer::element::U384};

    use super::*;

    type Fp12E = FieldElement<Degree12ExtensionField>;

    #[test]
    fn test_line_1() {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let expected = Fp12E::from_coefficients(&[
            "8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1",
            "0",
            "0",
            "0",
            "0",
            "0",
            "11bfe7fb8923ace27a0692871443365b3e5e0dd7ef388153c5bb27d923a3dedb6b9f9cbfd022a8eb913e281a830fac9c",
            "13d0fab25f0e7099c6ffaa8ddd3cb036f5c35df322a9359ff8f98a4f5c6a84c50b6007c296eafda2ffa2d20b253a6633",
            "4c208bdb300097927393e963768099390a3f9d581d8828070e39167384e44fdaf716fa49d68b0bdf431a2f53189c109",
            "546ca700477f9c2f9def9691be2f7e6eaa0f1474cb64c53ce3b4d4da03cbdac75933b5468ab4b88cf058f147ba2cda9",
            "0",
            "0"
        ]);
        assert_eq!(line(&g2, &g2, &g1), expected);
    }

    #[test]
    fn test_line_2() {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let g2_times_5 = g2.operate_with_self(5_u16).to_affine();
        let expected = Fp12E::from_coefficients(&[
            "8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1",
            "0",
            "0",
            "0",
            "0",
            "0",
            "c61af5cbff75e6cdd8100b22636ec410a1cb914a599775a29590dd2c48af1f18a71e120fb72ddc7c1ca57ce58a8670f",
            "4b7599ae8879affcb68f23ac23ddc7316f81013a2caa8925f33fe978f19ce00effd7e5bd6b51e2d9723e66fd897e125",
            "449f41bddfdc54476c92f4249f1ce75a9693eb8f0b34ba3c57de3edaa29ba069f0d97376eadb1be6c5a5dfdf595302b",
            "ebf57b77d5ffe1476a8c8c0f0e0e73e8d32f070c89f09e465cb39312463bc42768c6028a481fb8ba2f5d2a95cd4eedb",
            "0",
            "0"
        ]);
        assert_eq!(line(&g2, &g2_times_5, &g1), expected);
    }

    #[test]
    fn batch_ate_pairing_bilinearity() {
        let p = BLS12381Curve::generator().to_affine();
        let q = BLS12381TwistCurve::generator().to_affine();
        let a = U384::from_u64(11);
        let b = U384::from_u64(93);

        let result = batch_ate(&[
            (
                &p.operate_with_self(a).to_affine(),
                &q.operate_with_self(b).to_affine(),
            ),
            (&p.operate_with_self(a * b).to_affine(), &q.neg()),
        ]);
        assert_eq!(result, FieldElement::one());
    }
}
