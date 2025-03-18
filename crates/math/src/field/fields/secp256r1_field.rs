use crate::{
    field::fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    unsigned_integer::element::U256,
};

type Secp256r1MontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigSecp256r1PrimeField;
impl IsModulus<U256> for MontgomeryConfigSecp256r1PrimeField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff",
    );
}

pub type Secp256r1PrimeField =
    Secp256r1MontgomeryBackendPrimeField<MontgomeryConfigSecp256r1PrimeField>;
