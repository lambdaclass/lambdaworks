use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        //traits::IsFFTField,
    },
    unsigned_integer::element::UnsignedInteger
};

pub type U64 = UnsignedInteger<1>;
pub type U64PrimeField<T> = MontgomeryBackendPrimeField<T,1>;


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigBabybear31PrimeField;
impl IsModulus<U64> for MontgomeryConfigBabybear31PrimeField {
    //Babybear Prime p = 2^31 - 2^27 + 1 = 0x78000001
    const MODULUS: U64 = U64::from_u64(2013265921);
}

pub type Babybear31PrimeField = U64PrimeField<MontgomeryConfigBabybear31PrimeField>;

impl FieldElement<Babybear31PrimeField> {
    pub fn to_bytes_le(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        let mut bytes: [u8; 8] = [0; 8];

        for i in (0..4).rev() {
            let limb_bytes = limbs[i].to_le_bytes();
            for j in 0..8 {
                // i = 3 ->
                bytes[(3 - i) * 8 + j] = limb_bytes[j]
            }
        }
        bytes
    }

    pub fn to_bytes_be(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        let mut bytes: [u8; 8] = [0; 8];

        for i in 0..4 {
            let limb_bytes = limbs[i].to_be_bytes();
            for j in 0..8 {
                bytes[i * 8 + j] = limb_bytes[j]
            }
        }
        bytes
    }
}

impl PartialOrd for FieldElement<Babybear31PrimeField> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.representative().partial_cmp(&other.representative())
    }
}

impl Ord for FieldElement<Babybear31PrimeField> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.representative().cmp(&other.representative())
    }
}