use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::{UnsignedInteger, U256},
};

pub type PallasMontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigPallas255PrimeField;
impl IsModulus<U256> for MontgomeryConfigPallas255PrimeField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
    );
}
