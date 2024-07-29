use super::curve::MILLER_LOOP_CONSTANT;
// need to define MILLER_LOOP_CONSTANT in curve.rs
// we think that MILLER_LOOP_CONSTANT = 6x+2 = 29793968203157093288
// with x = 496566136719284888
// see https://hackmd.io/@Wimet/ry7z1Xj-2
// @Juan is this the same parameter used in the NAF representation? 
// t = 6x^2. Where x = 4965661367192848881

use super::{
    curve::BN254Curve,
    field_extension::{BN254PrimeField, Degree12ExtensionField, Degree2ExtensionField},
    twist::BN254TwistCurve,
};
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
// We have to find the SUBGROUP_ORDER and see where it's used.
// In the implementation of zksync we have:
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
// 30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001 (fist I coverted it into hex)
pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("TODO");

// Need implementation of NAF representation
// 
/// Millers loop uses to iterate the NAF representation of the MILLER_LOOP_CONSTANT
/// A NAF representation uses values: -1, 0 and 1. https://en.wikipedia.org/wiki/Non-adjacent_form.
pub const MILLER_CONSTANT_NAF: [i32; 115] = [
    1, 0, -1, 0, 0, -1, 0, -1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0,
    -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, -1, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0,
    0, 0, -1, 0, -1, 0, -1, 0, 0, -1, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0
];


pub struct BN254AtePairing;
impl IsPairing for BN254AtePairing {
    type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
    type OutputField = Degree12ExtensionField;
    // Compute the product of the ate pairings for a list of point pairs.
    // To optimize the pairing computation, we compute first all the miller
    // loops and multiply each other (so that we can then do the final expon).
    fn compute_batch(
        pairs: &[(&Self::G1Point, &Self::G2Point)],
    ) -> Result<FieldElement<Self::OutputField>, PairingError> {
        let mut result = FieldElement::one();
        for (p, q) in pairs {
            // We think we can remove the condition !p.is_in_subgroup() because
            // the subgroup oF G1 is G1 (see BN254 for the rest of us).
            // We have to implement is_in_subgroup() for the TwistesCurve in curve.rs
            if !p.is_in_subgroup() || !q.is_in_subgroup() {
                return Err(PairingError::PointNotInSubgroup);
            }
            if !p.is_neutral_element() && !q.is_neutral_element() {
                let p = p.to_affine();
                let q = q.to_affine();
                result *= miller(&q, &p);
            }
        }
        Ok(final_exponentiation(&result))
    }
}
// TODO
// We need a function that computes the double of a G2 point and its tangent line.
// In the implementation of bls381, this function also changes the t's and accumulator's (f) values.
// Initially t = Q, accumulator (f) = 1. See https://eprint.iacr.org/2010/354.pdf.
// Question: In https://eprint.iacr.org/2010/354.pdf P shouldn't be ProjectivePoint (Q is ok).
// incomplete

fn double_accumulate_line (
    t: &mut ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
)
{
    let [x_q,y_q,z_q] = t.coordinates();
    let [x_p,y_p,_] = p.coordinates();
    
    a = x_q.square(); // 1
    b = y_q.square(); // 2
    c = b.square();// 3
    d = (b+x_q).square()-a-c; // 4
    d = d + d; //2 temp3 // 5
    e = a + a + a;  // 3 temp1 // 6
    f = x_q + e; // 7
    g = f.square(); // 8
    x_t = f - d -d ; // 9
    z_t = (y_q + z_q).square() - b - z_q.square(); // 10
    y_t = (d-x_t)*e - c.double().double().double(); //11
    d =  (e*z_q.square()).double().neg();square();//12
    d= d*x_p; //13
    e = e.square() - a  - g - b -b -b -b;  //14
    a  = (z_t * z_q-square()).double(); //15
    a = a * y_p;// 16 
    let a_0 = a; //18
    let a_1 =  d + e *v ; //19

    let t = ShortWeierstrassProjectivePoint::new(x_t, y_t, z_t);
    let l = a_0 + a_1*w;
}

//TODO
// We need a function that computes the addition of two G2 points and the line through them.
fn add_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    q: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    // TODO
}