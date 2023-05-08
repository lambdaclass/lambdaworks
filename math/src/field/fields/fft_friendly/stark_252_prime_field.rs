use crate::{
    field::{
        fields::montgomery_backed_prime_fields::{IsModulus, U256PrimeField},
        traits::IsFFTField,
    },
    unsigned_integer::element::{UnsignedInteger, U256},
};

#[derive(Clone, Debug)]
pub struct MontgomeryConfigStark252PrimeField;
impl IsModulus<U256> for MontgomeryConfigStark252PrimeField {
    const MODULUS: U256 =
        U256::from("800000000000011000000000000000000000000000000000000000000000001");
}

pub type Stark252PrimeField = U256PrimeField<MontgomeryConfigStark252PrimeField>;

impl IsFFTField for Stark252PrimeField {
    const TWO_ADICITY: u64 = 48;
    // Change this line for a new function like `from_limbs`.
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: U256 = UnsignedInteger {
        limbs: [
            219038664817244121,
            2879838607450979157,
            15244050560987562958,
            16338897044258952332,
        ],
    };

    fn field_name() -> &'static str {
        "stark256"
    }
}
