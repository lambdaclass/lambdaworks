use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U64,
};

pub type U64MontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 1>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MontgomeryConfigMersenne31PrimeField;
impl IsModulus<U64> for MontgomeryConfigMersenne31PrimeField {
    //Mersenne Prime p = 2^31 - 1
    const MODULUS: U64 = U64::from_u64(2147483647);
}

pub type Mersenne31MontgomeryPrimeField =
    U64MontgomeryBackendPrimeField<MontgomeryConfigMersenne31PrimeField>;

impl FieldElement<Mersenne31MontgomeryPrimeField> {
    pub fn to_bytes_le(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_le_bytes()
    }

    pub fn to_bytes_be(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_be_bytes()
    }
}

impl PartialOrd for FieldElement<Mersenne31MontgomeryPrimeField> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.representative().partial_cmp(&other.representative())
    }
}

impl Ord for FieldElement<Mersenne31MontgomeryPrimeField> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.representative().cmp(&other.representative())
    }
}
