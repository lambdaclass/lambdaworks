use crate::field::{
    element::FieldElement,
    fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
};

use crate::unsigned_integer::element::U384;

pub const BN254_PRIME_FIELD_ORDER: U384 =
    U384::from_hex_unchecked("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");

// Fp_Bn254 (prime field)

#[derive(Clone, Debug)]
pub struct BN254FieldModulus;

impl IsModulus<U384> for BN254FieldModulus {
    const MODULUS: U384 = BN254_PRIME_FIELD_ORDER;
}

pub type BN254PrimeField = MontgomeryBackendPrimeField<BN254FieldModulus, 6>;

impl FieldElement<BN254PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U384::from_hex_unchecked(a_hex))
    }
}
