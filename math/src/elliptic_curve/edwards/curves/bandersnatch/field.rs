//! Base field of bandersantch -- which is also the scalar field of BLS12-381 curve.

use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U256,
};

pub const BANDERSNATCH_PRIME_FIELD_ORDER: U256 =
    U256::from_hex_unchecked("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

#[derive(Clone, Debug)]
pub struct FqConfig;

impl IsModulus<U256> for FqConfig {
    const MODULUS: U256 = BANDERSNATCH_PRIME_FIELD_ORDER;
}

pub type FqField = MontgomeryBackendPrimeField<FqConfig, 4>;

impl FieldElement<FqField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U256::from(a_hex))
    }
}
pub type FqElement = FieldElement<FqField>;
