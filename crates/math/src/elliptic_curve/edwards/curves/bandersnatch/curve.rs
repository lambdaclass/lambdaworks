//! Bandersnatch curve implementation.
//!
//! Bandersnatch is a twisted Edwards curve defined over the scalar field of BLS12-381.
//! Curve equation: ax² + y² = 1 + dx²y² where a = -5.
//!
//! References:
//! - [Bandersnatch paper](https://eprint.iacr.org/2021/1152)
//! - [Arkworks implementation](https://github.com/arkworks-rs/curves/tree/master/ed_on_bls12_381_bandersnatch)

use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::edwards::point::EdwardsProjectivePoint;
use crate::elliptic_curve::edwards::traits::IsEdwards;
use crate::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use crate::unsigned_integer::element::U256;

/// Bandersnatch base field is BLS12-381's scalar field (Fr).
pub type BandersnatchBaseField = FrField;

/// The cofactor of the Bandersnatch curve (h = 4).
pub const BANDERSNATCH_COFACTOR: u64 = 4;

/// The order of the prime-order subgroup.
/// r = 13108968793781547619861935127046491459309155893440570251786403306729687672801
pub const BANDERSNATCH_SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("1cfb69d4ca675f520cce760202687600ff8f87007419047174fd06b52876e7e1");

#[derive(Clone, Debug)]
pub struct BandersnatchCurve;

impl IsEllipticCurve for BandersnatchCurve {
    type BaseField = BandersnatchBaseField;
    type PointRepresentation = EdwardsProjectivePoint<Self>;

    /// Generator from [Arkworks](https://github.com/arkworks-rs/curves/blob/master/ed_on_bls12_381_bandersnatch/src/curves/mod.rs).
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::new(U256::from_hex_unchecked(
                "29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18",
            )),
            FieldElement::new(U256::from_hex_unchecked(
                "2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166",
            )),
            FieldElement::one(),
        ])
        .expect("valid generator")
    }
}

impl IsEdwards for BandersnatchCurve {
    /// a = -5 (mod p)
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::new(U256::from_hex_unchecked(
            "73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFEFFFFFFFC",
        ))
    }

    fn d() -> FieldElement<Self::BaseField> {
        FieldElement::new(U256::from_hex_unchecked(
            "6389C12633C267CBC66E3BF86BE3B6D8CB66677177E54F92B369F2F5188D58E7",
        ))
    }
}

impl EdwardsProjectivePoint<BandersnatchCurve> {
    /// Checks if the point is in the prime-order subgroup.
    pub fn is_in_subgroup(&self) -> bool {
        self.is_neutral_element()
            || self
                .operate_with_self(BANDERSNATCH_SUBGROUP_ORDER)
                .is_neutral_element()
    }

    /// Clears the cofactor by multiplying by h = 4.
    pub fn clear_cofactor(&self) -> Self {
        self.operate_with_self(BANDERSNATCH_COFACTOR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_valid_and_in_subgroup() {
        let g = BandersnatchCurve::generator();
        // Verify generator satisfies curve equation
        let g_affine = g.to_affine();
        assert_eq!(
            BandersnatchCurve::defining_equation(g_affine.x(), g_affine.y()),
            FieldElement::zero()
        );
        // Verify generator is in prime-order subgroup
        assert!(g.is_in_subgroup());
    }
}
