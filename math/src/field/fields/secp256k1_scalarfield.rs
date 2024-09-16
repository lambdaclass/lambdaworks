use crate::{
    field::fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    unsigned_integer::element::U256,
};

type Secp256k1MontgomeryBackendScalarField<T> = MontgomeryBackendPrimeField<T, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigSecp256k1ScalarField;
impl IsModulus<U256> for MontgomeryConfigSecp256k1ScalarField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
    );
}

pub type Secp256k1ScalarField =
    Secp256k1MontgomeryBackendScalarField<MontgomeryConfigSecp256k1ScalarField>;
