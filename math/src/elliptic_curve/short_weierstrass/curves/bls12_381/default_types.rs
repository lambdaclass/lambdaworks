use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        traits::IsFFTField,
    },
    unsigned_integer::element::{UnsignedInteger, U256},
};

#[derive(Clone, Debug)]
pub struct FrConfig;

/// Modulus of bls 12 381 subgroup
impl IsModulus<U256> for FrConfig {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    );
}

/// FrField using MontgomeryBackend for bls 12 381
pub type FrField = MontgomeryBackendPrimeField<FrConfig, 4>;
/// FrElement using MontgomeryBackend for bls 12 381
pub type FrElement = FieldElement<FrField>;

impl IsFFTField for FrField {
    const TWO_ADICITY: u64 = 32;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType = UnsignedInteger::from_hex_unchecked(
        "2ab00961a08a499d84dd396c349d9b3cc5e433d6fa78eb2b54cc39d9bb30bbb7",
    );
}
