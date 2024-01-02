use crate::{
    field::fields::montgomery_backed_prime_fields::IsModulus, unsigned_integer::element::U256,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigVesta255PrimeField;
impl IsModulus<U256> for MontgomeryConfigVesta255PrimeField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
    );
}
