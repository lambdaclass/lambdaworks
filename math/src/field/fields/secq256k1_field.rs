// TODO: Use secp256k1_scalarfield from the PR add_curves.
use crate::{
    field::fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    unsigned_integer::element::U256,
};

type Secq256k1MontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigSecq256k1PrimeField;
impl IsModulus<U256> for MontgomeryConfigSecq256k1PrimeField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
    );
}

pub type Secq256k1PrimeField =
    Secq256k1MontgomeryBackendPrimeField<MontgomeryConfigSecq256k1PrimeField>;
