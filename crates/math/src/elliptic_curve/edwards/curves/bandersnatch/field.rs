//! Base field of Bandersnatch curve (Fq).
//!
//! This is the scalar field of the BLS12-381 curve, which serves as the
//! base field for Bandersnatch. This field is used for the coordinates
//! of points on the curve.
//!
//! # Field Parameters
//!
//! - Prime modulus: p = 52435875175126190479447740508185965837690552500527637822603658699938581184513
//! - In hex: 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
//!
//! This is the same as the BLS12-381 scalar field (Fr in BLS12-381 terminology).

use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U256,
};

/// The prime modulus of the Bandersnatch base field.
/// p = 52435875175126190479447740508185965837690552500527637822603658699938581184513
/// This is the scalar field of BLS12-381.
pub const BANDERSNATCH_PRIME_FIELD_ORDER: U256 =
    U256::from_hex_unchecked("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

#[derive(Clone, Debug)]
pub struct FqConfig;

impl IsModulus<U256> for FqConfig {
    const MODULUS: U256 = BANDERSNATCH_PRIME_FIELD_ORDER;
}

/// Base field for Bandersnatch (Fq).
/// This is the field over which curve points' coordinates are defined.
pub type FqField = MontgomeryBackendPrimeField<FqConfig, 4>;

/// Base field element type alias.
pub type FqElement = FieldElement<FqField>;

impl FieldElement<FqField> {
    /// Creates a base field element from a hexadecimal string.
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U256::from(a_hex))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_field_order_is_correct() {
        // Verify the order matches the BLS12-381 scalar field
        let order = BANDERSNATCH_PRIME_FIELD_ORDER;
        assert_eq!(
            order,
            U256::from_hex_unchecked(
                "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
            )
        );
    }

    #[test]
    fn base_field_arithmetic_works() {
        let a = FqElement::from(5u64);
        let b = FqElement::from(3u64);
        let sum = &a + &b;
        assert_eq!(sum, FqElement::from(8u64));

        let product = &a * &b;
        assert_eq!(product, FqElement::from(15u64));
    }

    #[test]
    fn new_base_creates_element_from_hex() {
        let elem = FqElement::new_base("10");
        assert_eq!(elem, FqElement::from(16u64));
    }
}
