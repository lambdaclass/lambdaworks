use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U256,
};

#[derive(Clone, Debug)]
pub struct FrConfig;

/// Modulus of bls 12 381 subgroup
impl IsModulus<U256> for FrConfig {
    const MODULUS: U256 =
        U256::from_hex("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
}

/// FrField using MontgomeryBackend for bls 12 381
pub type FrField = MontgomeryBackendPrimeField<FrConfig, 4>;
/// FrElement using MontgomeryBackend for bls 12 381
pub type FrElement = FieldElement<FrField>;
