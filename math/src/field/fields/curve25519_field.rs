use crate::{
    field::fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    unsigned_integer::element::U256,
};

type VestaMontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 4>;

//  p = "0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed"
//  = 57896044618658097711785492504343953926634992332820282019728792003956564819949
//  = 2^255 -19
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigCurve25519PrimeField;
impl IsModulus<U256> for MontgomeryConfigCurve25519PrimeField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed",
    );
}

pub type Curve25519PrimeField =
    VestaMontgomeryBackendPrimeField<MontgomeryConfigCurve25519PrimeField>;
