use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, U64PrimeField},
        traits::IsFFTField,
    },
    unsigned_integer::element::{UnsignedInteger, U64},
};

#[derive(Clone, Debug, Hash, Copy)]
pub struct MontgomeryConfigStark101PrimeField;
impl IsModulus<U64> for MontgomeryConfigStark101PrimeField {
    /// 3 * 2^30 + 1
    const MODULUS: U64 = U64::from_hex_unchecked("c0000001");
}

pub type Stark101PrimeField = U64PrimeField<MontgomeryConfigStark101PrimeField>;

impl IsFFTField for Stark101PrimeField {
    const TWO_ADICITY: u64 = 30;

    const TWO_ADIC_PRIMITIVE_ROOT_OF_UNITY: U64 = UnsignedInteger::from_hex_unchecked("bb6e79d");

    fn field_name() -> &'static str {
        "stark101"
    }
}

impl FieldElement<Stark101PrimeField> {
    pub fn to_bytes_le(&self) -> [u8; 8] {
        let limbs = self.canonical().limbs;
        limbs[0].to_le_bytes()
    }

    pub fn to_bytes_be(&self) -> [u8; 8] {
        let limbs = self.canonical().limbs;
        limbs[0].to_be_bytes()
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for FieldElement<Stark101PrimeField> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.canonical().partial_cmp(&other.canonical())
    }
}

impl Ord for FieldElement<Stark101PrimeField> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.canonical().cmp(&other.canonical())
    }
}

#[cfg(test)]
mod test_stark101_prime_field {
    use super::Stark101PrimeField;
    use crate::{
        field::{element::FieldElement, traits::IsFFTField},
        traits::ByteConversion,
    };

    #[test]
    fn two_adic_order() {
        let w = FieldElement::<Stark101PrimeField>::from(
            Stark101PrimeField::TWO_ADIC_PRIMITIVE_ROOT_OF_UNITY.limbs[0],
        );

        assert_eq!(
            w.pow(1u64 << Stark101PrimeField::TWO_ADICITY),
            FieldElement::one()
        );
        assert_ne!(
            w.pow(1u64 << (Stark101PrimeField::TWO_ADICITY >> 1)),
            FieldElement::one()
        );
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
        let element = FieldElement::<Stark101PrimeField>::from_hex_unchecked("0123456701234567");
        let bytes = element.to_bytes_le();
        let expected_bytes: [u8; 8] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
        let element = FieldElement::<Stark101PrimeField>::from_hex_unchecked("0123456701234567");
        let bytes = element.to_bytes_be();
        let expected_bytes: [u8; 8] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]

    fn byte_serialization_and_deserialization_works_le() {
        let element = FieldElement::<Stark101PrimeField>::from_hex_unchecked("7654321076543210");
        let bytes = element.to_bytes_le();
        let from_bytes = FieldElement::<Stark101PrimeField>::from_bytes_le(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }

    #[test]

    fn byte_serialization_and_deserialization_works_be() {
        let element = FieldElement::<Stark101PrimeField>::from_hex_unchecked("7654321076543210");
        let bytes = element.to_bytes_be();
        let from_bytes = FieldElement::<Stark101PrimeField>::from_bytes_be(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }
}
