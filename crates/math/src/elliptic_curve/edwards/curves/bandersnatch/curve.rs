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
//! - Base field: BLS12-381 scalar field (Fq)
//! - Scalar field order (r): 13108968793781547619861935127046491459309155893440570251786403306729687672801
//! - Cofactor (h): 4
//! - Full curve order: h * r
//!
//! # References
//!
//! - [Bandersnatch paper](https://eprint.iacr.org/2021/1152)
//! - [Arkworks implementation](https://github.com/arkworks-rs/curves/tree/master/ed_on_bls12_381_bandersnatch)

pub use super::field::FqField;
pub use super::scalar_field::{FrElement, FrField, BANDERSNATCH_SUBGROUP_ORDER};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::edwards::point::EdwardsProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{elliptic_curve::edwards::traits::IsEdwards, field::element::FieldElement};

pub type BaseBandersnatchFieldElement = FqField;

/// The cofactor of the Bandersnatch curve.
/// The full curve order is h * r, where h = 4 and r is the prime subgroup order.
pub const BANDERSNATCH_COFACTOR: u64 = 4;

/// Cofactor inverse in the scalar field.
/// COFACTOR_INV = 4^(-1) mod r
/// Used for clearing the cofactor when needed.
pub const BANDERSNATCH_COFACTOR_INV: &str =
    "15a36d1f0f390d83992f51e89448deb286e14da630a5a7a2f74e83eaacb5c5e1";

#[derive(Clone, Debug)]
pub struct BandersnatchCurve;

impl IsEllipticCurve for BandersnatchCurve {
    type BaseField = BaseBandersnatchFieldElement;
    type PointRepresentation = EdwardsProjectivePoint<Self>;

    /// Returns the generator point of the Bandersnatch curve.
    ///
    /// The generator point is defined with coordinates `(x, y, 1)`, where `x` and `y`
    /// are precomputed constants that belong to the curve. This generator generates
    /// the prime-order subgroup of order r.
    ///
    /// # Safety
    ///
    /// - The generator values are taken from the [Arkworks implementation](https://github.com/arkworks-rs/curves/blob/5a41d7f27a703a7ea9c48512a4148443ec6c747e/ed_on_bls12_381_bandersnatch/src/curves/mod.rs#L120)
    ///   and have been converted to hexadecimal.
    /// - `unwrap()` does not panic because:
    ///   - The generator point is **known to be valid** on the curve.
    ///   - The function only uses **hardcoded** and **verified** constants.
    /// - This function should **never** be modified unless the new generator is fully verified.
    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator point coordinates (x, y) are taken from a well-tested,
        //   verified implementation.
        // - The constructor will only fail if the values are invalid, which is
        //   impossible given that they are constants taken from a trusted source.
        let point = Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base(
                "29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18",
            ),
            FieldElement::<Self::BaseField>::new_base(
                "2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166",
            ),
            FieldElement::one(),
        ]);
        point.unwrap()
    }
}

impl IsEdwards for BandersnatchCurve {
    /// Returns the `a` coefficient of the curve equation.
    /// a = -5 (mod p)
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base(
            "73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFEFFFFFFFC",
        )
    }

    /// Returns the `d` coefficient of the curve equation.
    /// d = 45022363124591815672509500913686876175488063829319466900776701791074614335719
    fn d() -> FieldElement<Self::BaseField> {
        FieldElement::<Self::BaseField>::new_base(
            "6389C12633C267CBC66E3BF86BE3B6D8CB66677177E54F92B369F2F5188D58E7",
        )
    }
}

impl EdwardsProjectivePoint<BandersnatchCurve> {
    /// Checks if the point is in the prime-order subgroup.
    ///
    /// For Bandersnatch with cofactor h = 4, a point P is in the prime-order
    /// subgroup if and only if [r]P = O (neutral element), where r is the
    /// subgroup order.
    ///
    /// This is equivalent to checking that [h]P ≠ O for a point not at infinity
    /// (ensuring the point is not in a smaller subgroup).
    pub fn is_in_subgroup(&self) -> bool {
        // The neutral element is always in the subgroup
        if self.is_neutral_element() {
            return true;
        }

        // A point is in the prime-order subgroup if multiplying by the subgroup
        // order gives the neutral element.
        self.operate_with_self(BANDERSNATCH_SUBGROUP_ORDER)
            .is_neutral_element()
    }

    /// Clears the cofactor by multiplying by h = 4.
    ///
    /// This maps any point on the curve to a point in the prime-order subgroup.
    /// For a point P, returns [h]P = [4]P.
    pub fn clear_cofactor(&self) -> Self {
        self.operate_with_self(BANDERSNATCH_COFACTOR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U256,
    };

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BaseBandersnatchFieldElement>;

    fn point_1() -> EdwardsProjectivePoint<BandersnatchCurve> {
        let x = FEE::new_base("29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18");
        let y = FEE::new_base("2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166");

        BandersnatchCurve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn test_scalar_mul() {
        let g = BandersnatchCurve::generator();
        let result1 = g.operate_with_self(5u16);

        assert_eq!(
            result1.x().clone(),
            FEE::new_base("68CBECE0B8FB55450410CBC058928A567EED293D168FAEF44BFDE25F943AABE0")
        );

        let scalar =
            U256::from_hex("1CFB69D4CA675F520CCE760202687600FF8F87007419047174FD06B52876E7E6")
                .unwrap();
        let result2 = g.operate_with_self(scalar);

        assert_eq!(
            result2.x().clone(),
            FEE::new_base("68CBECE0B8FB55450410CBC058928A567EED293D168FAEF44BFDE25F943AABE0")
        );
    }

    #[test]
    fn test_create_valid_point_works() {
        let p = BandersnatchCurve::generator();

        assert_eq!(p, p.clone());
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(
            *p.x(),
            FEE::new_base("29C132CC2C0B34C5743711777BBE42F32B79C022AD998465E1E71866A252AE18")
        );
        assert_eq!(
            *p.y(),
            FEE::new_base("2A6C669EDA123E0F157D8B50BADCD586358CAD81EEE464605E3167B6CC974166")
        );
        assert_eq!(*p.z(), FEE::new_base("1"));
    }

    #[test]
    fn create_invalid_points_panics() {
        assert_eq!(
            BandersnatchCurve::create_point_from_affine(FEE::from(1), FEE::from(1)).unwrap_err(),
            EllipticCurveError::InvalidPoint
        )
    }

    #[test]
    fn equality_works() {
        let g = BandersnatchCurve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BandersnatchCurve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn generator_is_in_subgroup() {
        let g = BandersnatchCurve::generator();
        assert!(g.is_in_subgroup());
    }

    #[test]
    fn arbitrary_point_is_in_subgroup() {
        let g = BandersnatchCurve::generator();
        let p = g.operate_with_self(12345u64);
        assert!(p.is_in_subgroup());
    }

    #[test]
    fn neutral_element_is_in_subgroup() {
        let neutral = EdwardsProjectivePoint::<BandersnatchCurve>::neutral_element();
        assert!(neutral.is_in_subgroup());
    }

    #[test]
    fn clear_cofactor_gives_subgroup_point() {
        let g = BandersnatchCurve::generator();
        let cleared = g.clear_cofactor();
        assert!(cleared.is_in_subgroup());
    }

    #[test]
    fn cofactor_is_correct() {
        assert_eq!(BANDERSNATCH_COFACTOR, 4);
    }

    #[test]
    fn generator_satisfies_curve_equation() {
        let g = BandersnatchCurve::generator().to_affine();
        let result = BandersnatchCurve::defining_equation(g.x(), g.y());
        assert_eq!(result, FEE::zero());
    }

    #[test]
    fn subgroup_order_times_generator_is_neutral() {
        let g = BandersnatchCurve::generator();
        let result = g.operate_with_self(BANDERSNATCH_SUBGROUP_ORDER);
        assert!(result.is_neutral_element());
    }

    #[test]
    fn test_negation() {
        let g = BandersnatchCurve::generator();
        let neg_g = g.neg();
        let sum = g.operate_with(&neg_g);
        assert!(sum.is_neutral_element());
    }

    #[test]
    fn test_double() {
        let g = BandersnatchCurve::generator();
        let double1 = g.operate_with(&g);
        let double2 = g.operate_with_self(2u16);
        assert_eq!(double1, double2);
    }
}
