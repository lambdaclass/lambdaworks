use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U256,
};

pub type PallasMontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigPallas255PrimeField;
impl IsModulus<U256> for MontgomeryConfigPallas255PrimeField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
    );
}

pub type Pallas255PrimeField =
    PallasMontgomeryBackendPrimeField<MontgomeryConfigPallas255PrimeField>;

impl FieldElement<Pallas255PrimeField> {
    pub fn to_bytes_le(&self) -> [u8; 32] {
        let limbs = self.representative().limbs;
        let mut bytes: [u8; 32] = [0; 32];

        for i in (0..4).rev() {
            let limb_bytes = limbs[i].to_le_bytes();
            for j in 0..8 {
                // i = 3 ->
                bytes[(3 - i) * 8 + j] = limb_bytes[j]
            }
        }
        bytes
    }

    pub fn to_bits_le(&self) -> [bool; 256] {
        let limbs = self.representative().limbs;
        let mut bits = [false; 256];

        for i in (0..4).rev() {
            let limb_bytes = limbs[i].to_le_bytes();
            let limb_bytes_starting_index = (3 - i) * 8;
            for (j, byte) in limb_bytes.iter().enumerate() {
                let byte_index = (limb_bytes_starting_index + j) * 8;
                for k in 0..8 {
                    let bit_index = byte_index + k;
                    let bit_value = (byte >> k) & 1 == 1;
                    bits[bit_index] = bit_value;
                }
            }
        }
        bits
    }

    pub fn to_bytes_be(&self) -> [u8; 32] {
        let limbs = self.representative().limbs;
        let mut bytes: [u8; 32] = [0; 32];

        for i in 0..4 {
            let limb_bytes = limbs[i].to_be_bytes();
            for j in 0..8 {
                bytes[i * 8 + j] = limb_bytes[j]
            }
        }
        bytes
    }
}

#[cfg(test)]
mod test_pallas_255_bytes_ops {
    use super::Pallas255PrimeField;
    use crate::{field::element::FieldElement, traits::ByteConversion};

    #[test]
    #[cfg(feature = "std")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
        let element = FieldElement::<Pallas255PrimeField>::from_hex_unchecked(
            "\
            0123456701234567\
            0123456701234567\
            0123456701234567\
            0123456701234567\
        ",
        );
        let bytes = element.to_bytes_le();
        let expected_bytes: [u8; 32] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]
    #[cfg(feature = "std")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
        let element = FieldElement::<Pallas255PrimeField>::from_hex_unchecked(
            "\
            0123456701234567\
            0123456701234567\
            0123456701234567\
            0123456701234567\
        ",
        );
        let bytes = element.to_bytes_be();
        let expected_bytes: [u8; 32] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]

    fn byte_serialization_and_deserialization_works_le() {
        let element = FieldElement::<Pallas255PrimeField>::from_hex_unchecked(
            "\
            0123456701234567\
            7654321076543210\
            7654321076543210\
            7654321076543210\
        ",
        );
        let bytes = element.to_bytes_le();
        let from_bytes = FieldElement::<Pallas255PrimeField>::from_bytes_le(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }

    #[test]

    fn byte_serialization_and_deserialization_works_be() {
        let element = FieldElement::<Pallas255PrimeField>::from_hex_unchecked(
            "\
            0123456701234567\
            7654321076543210\
            7654321076543210\
            7654321076543210\
        ",
        );
        let bytes = element.to_bytes_be();
        let from_bytes = FieldElement::<Pallas255PrimeField>::from_bytes_be(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }
}
