use crate::{
    field::fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    unsigned_integer::element::U256,
};

type Secp256k1MontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigSecp256k1PrimeField;
impl IsModulus<U256> for MontgomeryConfigSecp256k1PrimeField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
    );
}

pub type Secp256k1PrimeField =
    Secp256k1MontgomeryBackendPrimeField<MontgomeryConfigSecp256k1PrimeField>;
