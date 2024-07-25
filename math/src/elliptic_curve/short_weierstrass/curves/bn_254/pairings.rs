use super::curve::MILLER_LOOP_CONSTANT;
// need to define MILLER_LOOP_CONSTANT in curve.rs
// we think that MILLER_LOOP_CONSTANT = 6x+2 = 29793968203157093288
// with x = 496566136719284888
// see https://hackmd.io/@Wimet/ry7z1Xj-2

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
pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("TODO");

// Need implementation of NAF representation


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




// 


