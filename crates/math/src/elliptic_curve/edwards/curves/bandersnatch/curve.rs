//! Bandersnatch curve implementation.
//!
//! Bandersnatch is a twisted Edwards curve defined over the scalar field of BLS12-381.
//! It was designed for efficient zero-knowledge proofs and is used in Ethereum's
//! Verkle tree proposal.
//!
//! # Curve Equation
//!
//! The curve is defined by: ax² + y² = 1 + dx²y²
//!
//! where:
//! - a = -5
//! - d = 45022363124591815672509500913686876175488063829319466900776701791074614335719
//!
//! # Parameters
//!
//! - Base field: BLS12-381 scalar field (Fr) - reused from `bls12_381::default_types::FrField`
//! - Scalar field order (r): 13108968793781547619861935127046491459309155893440570251786403306729687672801
//! - Cofactor (h): 4
//! - Full curve order: h * r
//!
//! # References
//!
//! - [Bandersnatch paper](https://eprint.iacr.org/2021/1152)
//! - [Arkworks implementation](https://github.com/arkworks-rs/curves/tree/master/ed_on_bls12_381_bandersnatch)

pub use super::scalar_field::BANDERSNATCH_SUBGROUP_ORDER;
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::edwards::point::EdwardsProjectivePoint;
use crate::elliptic_curve::edwards::traits::IsEdwards;
use crate::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use crate::unsigned_integer::element::U256;

/// Bandersnatch base field is BLS12-381's scalar field (Fr).
pub type BandersnatchBaseField = FrField;

/// The cofactor of the Bandersnatch curve.
/// The full curve order is h * r, where h = 4 and r is the prime subgroup order.
pub const BANDERSNATCH_COFACTOR: u64 = 4;

#[derive(Clone, Debug)]
pub struct BandersnatchCurve;

impl IsEllipticCurve for BandersnatchCurve {
    type BaseField = BandersnatchBaseField;
    type PointRepresentation = EdwardsProjectivePoint<Self>;

    /// Returns the generator point of the Bandersnatch curve.
    ///
    /// This generator generates the prime-order subgroup of order r.
    ///
    /// Generator coordinates from [Arkworks](https://github.com/arkworks-rs/curves/blob/master/ed_on_bls12_381_bandersnatch/src/curves/mod.rs).
    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new(U256::from_hex_unchecked(
                "29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18",
            )),
            FieldElement::<Self::BaseField>::new(U256::from_hex_unchecked(
                "2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166",
            )),
            FieldElement::one(),
        ])
        .expect("Generator point is valid")
    }
}

impl IsEdwards for BandersnatchCurve {
    /// Returns the `a` coefficient: a = -5 (mod p)
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new(U256::from_hex_unchecked(
            "73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFEFFFFFFFC",
        ))
    }

    /// Returns the `d` coefficient.
    /// d = 45022363124591815672509500913686876175488063829319466900776701791074614335719
    fn d() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new(U256::from_hex_unchecked(
            "6389C12633C267CBC66E3BF86BE3B6D8CB66677177E54F92B369F2F5188D58E7",
        ))
    }
}

impl EdwardsProjectivePoint<BandersnatchCurve> {
    /// Checks if the point is in the prime-order subgroup.
    ///
    /// For Bandersnatch with cofactor h = 4, a point P is in the prime-order
    /// subgroup if and only if [r]P = O (neutral element).
    pub fn is_in_subgroup(&self) -> bool {
        if self.is_neutral_element() {
            return true;
        }
        self.operate_with_self(BANDERSNATCH_SUBGROUP_ORDER)
            .is_neutral_element()
    }

    /// Clears the cofactor by multiplying by h = 4.
    ///
    /// Maps any curve point to the prime-order subgroup.
    pub fn clear_cofactor(&self) -> Self {
        self.operate_with_self(BANDERSNATCH_COFACTOR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cyclic_group::IsGroup;
    use crate::elliptic_curve::traits::EllipticCurveError;

    type FE = FieldElement<BandersnatchBaseField>;

    #[test]
    fn generator_satisfies_curve_equation() {
        let g = BandersnatchCurve::generator().to_affine();
        assert_eq!(
            BandersnatchCurve::defining_equation(g.x(), g.y()),
            FE::zero()
        );
    }

    #[test]
    fn generator_is_in_subgroup() {
        assert!(BandersnatchCurve::generator().is_in_subgroup());
    }

    #[test]
    fn subgroup_order_times_generator_is_neutral() {
        let g = BandersnatchCurve::generator();
        assert!(g
            .operate_with_self(BANDERSNATCH_SUBGROUP_ORDER)
            .is_neutral_element());
    }

    #[test]
    fn scalar_mul_works() {
        let g = BandersnatchCurve::generator();
        let g5 = g.operate_with_self(5u16);
        let g3_plus_g2 = g
            .operate_with_self(3u16)
            .operate_with(&g.operate_with_self(2u16));
        assert_eq!(g5, g3_plus_g2);
    }

    #[test]
    fn invalid_point_rejected() {
        assert_eq!(
            BandersnatchCurve::create_point_from_affine(FE::from(1), FE::from(1)).unwrap_err(),
            EllipticCurveError::InvalidPoint
        );
    }

    #[test]
    fn negation_works() {
        let g = BandersnatchCurve::generator();
        assert!(g.operate_with(&g.neg()).is_neutral_element());
    }
}
