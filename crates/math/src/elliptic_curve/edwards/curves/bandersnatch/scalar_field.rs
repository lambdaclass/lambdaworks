//! Scalar field of Bandersnatch curve (Fr).
//!
//! This is the order of the prime-order subgroup, used for scalar multiplication.
//! The full curve has order h * r where h = 4 (cofactor) and r is this field's order.
//!
//! r = 13108968793781547619861935127046491459309155893440570251786403306729687672801

use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U256,
};

/// The order of the prime-order subgroup of Bandersnatch.
/// r = 13108968793781547619861935127046491459309155893440570251786403306729687672801
/// In hex: 0x1cfb69d4ca675f520cce760202687600ff8f87007419047174fd06b52876e7e1
pub const BANDERSNATCH_SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("1cfb69d4ca675f520cce760202687600ff8f87007419047174fd06b52876e7e1");

#[derive(Clone, Debug)]
pub struct FrConfig;

impl IsModulus<U256> for FrConfig {
    const MODULUS: U256 = BANDERSNATCH_SUBGROUP_ORDER;
}

/// Scalar field for Bandersnatch (Fr).
/// Used for scalar multiplication and operations in the prime-order subgroup.
pub type FrField = MontgomeryBackendPrimeField<FrConfig, 4>;

/// Scalar field element type alias.
pub type FrElement = FieldElement<FrField>;

impl FieldElement<FrField> {
    /// Creates a scalar field element from a hexadecimal string.
    pub fn new_scalar(a_hex: &str) -> Self {
        Self::new(U256::from(a_hex))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_field_order_is_correct() {
        // Verify the order matches the expected value
        let order = BANDERSNATCH_SUBGROUP_ORDER;
        assert_eq!(
            order,
            U256::from_hex_unchecked(
                "1cfb69d4ca675f520cce760202687600ff8f87007419047174fd06b52876e7e1"
            )
        );
    }

    #[test]
    fn scalar_field_arithmetic_works() {
        let a = FrElement::from(5u64);
        let b = FrElement::from(3u64);
        let sum = &a + &b;
        assert_eq!(sum, FrElement::from(8u64));
    }
}
