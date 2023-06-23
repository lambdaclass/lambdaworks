use crate::{
    field::{
        fields::montgomery_backed_prime_fields::{IsModulus, U256PrimeField},
        traits::IsFFTField, element::FieldElement,
    },
    unsigned_integer::element::{UnsignedInteger, U256},
};

#[derive(Clone, Debug)]
pub struct MontgomeryConfigStark252PrimeField;
impl IsModulus<U256> for MontgomeryConfigStark252PrimeField {
    const MODULUS: U256 =
        U256::from_hex_unchecked("800000000000011000000000000000000000000000000000000000000000001");
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

impl FieldElement<Stark252PrimeField> {
    pub fn to_bytes_be(&self) -> [u8;32] {
        let limbs = self.representative().limbs;
        let mut bytes: [u8;32] = [0;32];

        for i in 0..4 {
            let limb_bytes = limbs[i].to_be_bytes();
            for j in 0..8{
                bytes[i*8+j] = limb_bytes[j]
            }
        }
        bytes
    }
}


#[cfg(test)]
mod test_stark_252_bytes_ops {
    use crate::field::element::FieldElement;
    use super::Stark252PrimeField;


    #[test]
    fn bytes_be() {
        let one = FieldElement::<Stark252PrimeField>::one();
        let bytes = one.to_bytes_be();
        // let result = FieldElement::<Stark252PrimeField>::from_bytes_be(&bytes).unwrap();
        // assert_eq!(one,result);
    }
}
