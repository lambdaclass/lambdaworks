use std::mem::zeroed;

use rayon::iter::once;

use super::{
    curve::BN254Curve,
    field_extension::{BN254PrimeField, Degree12ExtensionField, Degree2ExtensionField},
    twist::BN254TwistCurve,
};
use crate::elliptic_curve::traits::FromAffine;
use crate::{cyclic_group::IsGroup, elliptic_curve::traits::IsPairing, errors::PairingError};
use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bn_254::field_extension::{Degree6ExtensionField, LevelTwoResidue},
        point::ShortWeierstrassProjectivePoint,
        traits::IsShortWeierstrass,
    },
    field::{element::FieldElement, extensions::cubic::HasCubicNonResidue},
    unsigned_integer::element::{UnsignedInteger, U256},
};


type FpE = FieldElement<BN254PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp6E = FieldElement<Degree6ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;

// Constants of BN254Curve:

// See https://hackmd.io/@jpw/bn254#Barreto-Naehrig-curves
// see the repo https://github.com/hecmas/zkNotebook/blob/main/src/BN254/constants.ts

// x = 4965661367192848881
// p = (36x)^4 + (36x)^ 3 + (24 x)^ 2 + 6x + 1
// t = 147946756881789318990833708069417712967 = (6x)^2 + 1 
// t is the trace of Frobenius.
// r = p + 1 - t = 36x^4 + 36x^3 + 18x^2 + 6x+ 1 = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// r is the number of points in BN254Curve.

pub const X: u64 = 0x44e992b44a6909f1;
pub const R: U256 = U256::from_hex_unchecked("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

// Constant used in the Miller Loop.
// the same that arkworks uses ATE_LOOP_COUNT.
// MILLER_CONSTANT = 6x + 2 = 29793968203157093288.
// Note that this is a representation using {1, -1, 0}, but it isn't a NAF representation
// because it has non zero values adjacent.
pub const MILLER_NAF_2: [i32; 65] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0, 0, 1, 1, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    1, 1, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 1, 1, 0, -1, 0,
    0, 1, 0, 1, 1,
];

// GAMMA constants used to compute the Frobenius morphism.
// We took these constants from https://github.com/hecmas/zkNotebook/blob/main/src/BN254/constants.ts#L48
// note for future self , we should use const_from_raw instead of new.

//  GAMMA_1i = (9 + u)^{i(p-1) / 6} for all i = 1..5
pub const GAMMA_12: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("2FB347984F7911F74C0BEC3CF559B143B78CC310C2C3330C99E39557176F553D"),
    FpE::from_hex_unchecked("16C9E55061EBAE204BA4CC8BD75A079432AE2A1D0B7C9DCE1665D51C640FCBA2"),
]);

pub const GAMMA_13: Fp2E = Fp2E::const_from_raw([
    FpE::from_hex_unchecked("63CF305489AF5DCDC5EC698B6E2F9B9DBAAE0EDA9C95998DC54014671A0135A"),
    FpE::from_hex_unchecked("7C03CBCAC41049A0704B5A7EC796F2B21807DC98FA25BD282D37F632623B0E3"),
]);


pub struct BN254AtePairing;
impl IsPairing for BN254AtePairing {
    type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
    type OutputField = Degree12ExtensionField;

    // Computes the product of the ate pairings for a list of point pairs.
    // To optimize the pairing computation, we compute first all the miller
    // loops and multiply each other (so that we can then do the final exponentiation).
    fn compute_batch(
        pairs: &[(&Self::G1Point, &Self::G2Point)],
    ) -> Result<FieldElement<Self::OutputField>, PairingError> {
        let mut result = Fp12E::one();
        for (p, q) in pairs {
            // We think we can remove the condition !p.is_in_subgroup() because
            // the subgroup oF G1 is G1 (see BN254 for the rest of us).
            if !q.is_in_subgroup() {
                return Err(PairingError::PointNotInSubgroup);
            }
            if !p.is_neutral_element() && !q.is_neutral_element() {
                result *= miller_naive(p, q);
            }
        }
        Ok(final_exponentiation_naive(result))
    }
}

// Computes Miller loop using oprate_with(), operate_with_self() and line_2().
/// See https://eprint.iacr.org/2010/354.pdf, page 4.
fn miller_naive(
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
) -> Fp12E {
    let mut t = q.clone();
    let mut f = Fp12E::from(1);
    let miller_length = MILLER_NAF_2.len();

    for i in (0..miller_length - 1).rev() {
        f = f.square() * line_2(&p, &t, &t);
        t = t.operate_with_self(2usize);

        if MILLER_NAF_2[i] == -1 {
            f = f * line_2(&p, &q.neg(), &t);
            t = t.operate_with(&q.neg());
        } else if MILLER_NAF_2[i] == 1 {
            f = f * line_2(&p, &q, &t);
            t = t.operate_with(&q);
        }
    }

    let [x_q, y_q, _] = q.to_affine().coordinates().clone();

    // q1 = ((x_q)^p, (y_q)^p).
    let q1 = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
        GAMMA_12 * x_q.conjugate(),
        GAMMA_13 * y_q.conjugate(),
        Fp2E::one(),
    ]);

    f = f * line_2(&p, &q1, &t);
    t = t.operate_with(&q1);

    let [x_q1, y_q1, _] = q1.to_affine().coordinates().clone();

    // q2 = ((x_q1)^p, (y_q1)^p).
    let q2 = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
        GAMMA_12 * x_q1.conjugate(),
        GAMMA_13 * y_q1.conjugate(),
        Fp2E::one(),
    ]);

    f = f * line_2(&p, &q2.neg(), &t);

    f
}


// Computes the line between q and t and evaluates it in p.
// Algorithm from this post: https://hackmd.io/@Wimet/ry7z1Xj-2#Line-Equations
// and this implementation: https://github.com/hecmas/zkNotebook/blob/main/src/BN254/common.ts#L7
fn line_2(
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    t: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
) -> (Fp12E) {
    /* TO DO?
    if p == FpE::zero() || q == Fp12E::zero() || t == Fp12E::zero() {
        ERROR
    } */

    let [x_p, y_p, _] = p.to_affine().coordinates().clone();
    let [x_q, y_q, _] = q.phi_inv().coordinates().clone();
    let [x_t, y_t, _] = t.phi_inv().coordinates().clone();

    //TODO: if p, q or t are inf return error.

    let mut l: Fp12E = Fp12E::from(1);

    // First case:
    if x_t != x_q {

        let a = y_p * (&x_q - &x_t).square();
        let b = x_p * (&y_t - &y_q).square();
        let c = (x_t * y_q - x_q * y_t).square();

        l = Fp12E::from_coefficients(&[
            "0",
            "0",
            &(a.value()[0].to_hex()),
            &(a.value()[1].to_hex()),
            "0",
            "0",
            "0",
            "0",
            &(b.value()[0].to_hex()),
            &(b.value()[1]).to_hex(),
            &(c.value()[0].to_hex()),
            &(c.value()[1]).to_hex(),
        ]);
    // Second case: t and q are the same points
    } else if y_t == y_q {

        let a = Fp2E::new([FpE::from(9), FpE::one()])
            * (x_t.pow(3 as u32).double() + x_t.pow(3 as u32) - y_t.square().double()); //(9 + u) * (3 * (x_t)^3 - 2 * (y_t)^2)
        let b = (y_p * y_t).double(); // 2 * y_t * y_p
        let c = -((&x_p * x_t.square()).double() + (x_p * x_q.square())); // -3 * x_p * (x_t)^2
        l = Fp12E::from_coefficients(&[
            &(a.value()[0].to_hex()),
            &(a.value()[1].to_hex()),
            "0",
            "0",
            &(c.value()[0].to_hex()),
            &(c.value()[1]).to_hex(),
            "0",
            "0",
            &(b.value()[0].to_hex()),
            &(b.value()[1]).to_hex(),
            "0",
            "0",
        ]);
    }

    l
}

// f ^ {(p^12 -1)/r} = (f^p12 * f^{-1})^{-r} = (f * f.inv())^{-r}
fn final_exponentiation_naive(f: Fp12E) -> Fp12E {
    ((&f * f.inv().unwrap()).pow(R)).inv().unwrap()
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve,
        unsigned_integer::element::U384,
    };

    use super::*;
    
    #[test]
    fn the_line_t_q_is_the_same_as_the_line_q_t() {
        let p = BN254Curve::generator();

        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1a031c43dfaa2dd04a2c5b2dd257b449ce088dfd6d8ca041f19365b94ae7ae0",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x19f3b18b9baad6dadea895c76728c461e7f188f1a3da94a697d90428f554f039",
                )),
            ]),
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1467f6d823536b43c1d13f7ce580cc56ba88ad999b12e27e355c114d819bee81",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x2f992ff71d0d08f6271f2d40039924e831d53a43a4772e4322710ee41daed756",
                )),
            ]),
            FieldElement::one(),
        ]);

        let t = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1c0cdfa0b0eb6dcd93968a84ff1f5dfec3284b588e4bf72e6ed01052c23e1058",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x57182543d53551f8dc42e18396f78bec25a36dd07de7006308f90af112fa5d0",
                )),
            ]),
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::new(U256::from_hex_unchecked(
                    "0x279b356bf1390a747e0d361cd896e31898f9d9705942c176c79aa904b5c89243",
                )),
                FieldElement::new(U256::from_hex_unchecked(
                    "0x1f537cf9f716088e6a4bf2779cd7e66f83a29a498d2476a9e5a8d7eb9c21967d",
                )),
            ]),
            FieldElement::one(),
        ]);

        assert_eq!(line_2(&p, &q, &t), line_2(&p, &t, &q));
    }
    
    // Computes pi(q) = ((x_q)^p, (y_q)^p)
    pub fn point_power_p(q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>) 
    -> ShortWeierstrassProjectivePoint<BN254TwistCurve> {
    let [x_q, y_q, _] = q.to_affine().coordinates().clone();
    ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
        GAMMA_12 * x_q.conjugate(),
        GAMMA_13 * y_q.conjugate(),
        Fp2E::one(),
    ])
}
    #[test]
    fn apply_12_times_point_power_p_is_identity() {
        let q = BN254TwistCurve::generator();
        let mut result = point_power_p(&q);
        for _ in 1..12 {
            result = point_power_p(&result);
        }
        assert_eq!(q, result)
    }
  
    #[test]
    // e(ap, bq) = e(abp, q) = e(p, abq) = e(bp, aq)
    fn batch_ate_pairing_bilinearity() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let a = U384::from_u64(11);
        let b = U384::from_u64(93);

        let result_1 = BN254AtePairing::compute_batch(&[
            (&p.operate_with_self(a), &q.operate_with_self(b)),
            (&p.operate_with_self(a * b), &q.neg()),
        ])
        .unwrap();
        
        let result_2 = BN254AtePairing::compute_batch(&[
            (&p.operate_with_self(a), &q.operate_with_self(b)),
            (&p.neg(), &q.operate_with_self(a * b)),
        ])
        .unwrap();

        let result_3 = BN254AtePairing::compute_batch(&[
            (&p.operate_with_self(a), &q.operate_with_self(b)),
            (&p.operate_with_self(a), &q.operate_with_self(b).neg()),
        ])
        .unwrap();
        assert_eq!(result_1, Fp12E::one());
        assert_eq!(result_2, Fp12E::one());
        assert_eq!(result_3, Fp12E::one());
    }

    #[test]
    fn ate_pairing_returns_one_when_one_element_is_the_neutral_element() {
        let p = BN254Curve::generator();
        let q = ShortWeierstrassProjectivePoint::neutral_element();
        let result = BN254AtePairing::compute_batch(&[(&p, &q)]).unwrap();
        assert_eq!(result, FieldElement::one());

        let p = ShortWeierstrassProjectivePoint::neutral_element();
        let q = BN254TwistCurve::generator();
        let result = BN254AtePairing::compute_batch(&[(&p, &q)]).unwrap();
        assert_eq!(result, Fp12E::one());
    }
 
    #[test]
    fn ate_pairing_errors_when_g2_element_is_not_in_subgroup() {
        let p = BN254Curve::generator();
        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edaddde46bd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920daef312c20b9f1099ecefa8b45575d349b0a6f04c16d0d58af900",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "22376289c558493c1d6cc413a5f07dcb54526a964e4e687b65a881aa9752faa2",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "05a7a5759338c23ca603c1c4adf979e004c2f3e3c5bad6f07693c59a85d600a9",
                )),
            ]),
            Fp2E::one(),
        ]);
        let result = BN254AtePairing::compute_batch(&[(&p, &q)]);
        assert!(result.is_err())
    } 

    #[test]
    fn ate_pairing_errors_when_g1_point_is_zero() {
        let p = ShortWeierstrassProjectivePoint::<BN254Curve>::new([
            FpE::zero(),
            FpE::zero(),
            FpE::one(),
        ]);
        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edaddde46bd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920daef312c20b9f1099ecefa8b45575d349b0a6f04c16d0d58af900",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "22376289c558493c1d6cc413a5f07dcb54526a964e4e687b65a881aa9752faa2",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "05a7a5759338c23ca603c1c4adf979e004c2f3e3c5bad6f07693c59a85d600a9",
                )),
            ]),
            Fp2E::one(),
        ]);
        let result = BN254AtePairing::compute_batch(&[(&p, &q)]);
        assert!(result.is_err())
    } 
}
