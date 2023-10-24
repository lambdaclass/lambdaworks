use super::{
    curve::BN254Curve,
    field_extension::{Degree12ExtensionField, Degree2ExtensionField},
    twist::BN254TwistCurve,
};
use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bn_254::field_extension::{Degree6ExtensionField, LevelTwoResidue},
            point::ShortWeierstrassProjectivePoint,
            traits::IsShortWeierstrass,
        },
        traits::IsPairing,
    },
    field::{element::FieldElement, extensions::cubic::HasCubicNonResidue},
    unsigned_integer::element::UnsignedInteger,
};

#[derive(Clone)]
pub struct BN254AtePairing;
impl IsPairing for BN254AtePairing {
    type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
    type OutputField = Degree12ExtensionField;

    /// Compute the product of the ate pairings for a list of point pairs.
    fn compute_batch(
        pairs: &[(&Self::G1Point, &Self::G2Point)],
    ) -> FieldElement<Self::OutputField> {
        let mut result = FieldElement::one();
        for (p, q) in pairs {
            if !p.is_neutral_element() && !q.is_neutral_element() {
                let p = p.to_affine();
                let q = q.to_affine();
                result = result * miller(&q, &p);
            }
        }
        final_exponentiation(&result)
    }
}

/// This is equal to the frobenius trace of the BN 254 curve minus one.
const MILLER_LOOP_CONSTANT: u64 = 0xFFFFFFFFF;

#[allow(unused)]
fn double_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
}

#[allow(unused)]
fn add_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
}

#[allow(unused)]
fn miller(
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
) -> FieldElement<Degree12ExtensionField> {
    FieldElement::one()
}

#[allow(unused)]
fn final_exponentiation(
    base: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    FieldElement::one()
}
