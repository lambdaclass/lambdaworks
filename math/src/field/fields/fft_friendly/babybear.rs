use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U64,
};

pub type U64MontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 1>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigBabybear31PrimeField;
impl IsModulus<U64> for MontgomeryConfigBabybear31PrimeField {
    //Babybear Prime p = 2^31 - 2^27 + 1 = 0x78000001
    const MODULUS: U64 = U64::from_u64(2013265921);
}

pub type Babybear31PrimeField =
    U64MontgomeryBackendPrimeField<MontgomeryConfigBabybear31PrimeField>;

impl FieldElement<Babybear31PrimeField> {
    pub fn to_bytes_le(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_le_bytes()
    }

    pub fn to_bytes_be(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_be_bytes()
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

#[cfg(test)]
mod test_babybear_31_bytes_ops {
    use super::Babybear31PrimeField;
    use crate::{field::element::FieldElement, traits::ByteConversion};

    #[test]
    #[cfg(feature = "std")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
        let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked(
            "\
            0123456701234567\
        ",
        );
        let bytes = element.to_bytes_le();
        let expected_bytes: [u8; 8] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]
    #[cfg(feature = "std")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
        let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked(
            "\
            0123456701234567\
        ",
        );
        let bytes = element.to_bytes_be();
        let expected_bytes: [u8; 8] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]

    fn byte_serialization_and_deserialization_works_le() {
        let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked(
            "\
            7654321076543210\
        ",
        );
        let bytes = element.to_bytes_le();
        let from_bytes = FieldElement::<Babybear31PrimeField>::from_bytes_le(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }

    #[test]

    fn byte_serialization_and_deserialization_works_be() {
        let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked(
            "\
            7654321076543210\
        ",
        );
        let bytes = element.to_bytes_be();
        let from_bytes = FieldElement::<Babybear31PrimeField>::from_bytes_be(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }
}
