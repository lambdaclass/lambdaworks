use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsPrimeField},
    },
    traits::ByteConversion,
};

/// Uses randomness from the transcript to create a FieldElement
/// One bit less than the max used by the FieldElement is used as randomness. For StarkFields, this would be 251 bits randomness.
/// Randomness is interpreted as limbs in BigEndian, and each Limb is ordered in BigEndian
pub fn transcript_to_field<F: IsPrimeField, T: Transcript>(transcript: &mut T) -> FieldElement<F>
where
    FieldElement<F>: lambdaworks_math::traits::ByteConversion,
{
    let mut randomness = transcript.challenge();
    randomness_to_field(&mut randomness)
}

/// Transforms some random bytes to a field
/// Slicing the randomness to one bit less than what the max number of the field is to ensure each random element has the same probability of appearing
fn randomness_to_field<F: IsPrimeField>(randomness: &mut [u8; 32]) -> FieldElement<F>
where
    FieldElement<F>: ByteConversion,
{
    let random_bits_required = F::field_bit_size() - 1;
    let random_bits_created = randomness.len() * 8;
    let mut bits_to_clear = random_bits_created - random_bits_required;

    let mut i = 0;
    while bits_to_clear >= 8 {
        randomness[i] = 0;
        bits_to_clear -= 8;
        i += 1;
    }

    let pre_mask: u8 = 1u8.checked_shl(8 - bits_to_clear as u32).unwrap_or(0);
    let mask: u8 = pre_mask.wrapping_sub(1);
    randomness[i] &= mask;

    FieldElement::from_bytes_be(randomness).unwrap()
}

pub fn transcript_to_usize<T: Transcript>(transcript: &mut T) -> usize {
    const CANT_BYTES_USIZE: usize = (usize::BITS / 8) as usize;
    let value = transcript.challenge()[..CANT_BYTES_USIZE]
        .try_into()
        .unwrap();
    usize::from_be_bytes(value)
}

pub fn sample_z_ood<F: IsPrimeField, T: Transcript>(
    lde_roots_of_unity_coset: &[FieldElement<F>],
    trace_roots_of_unity: &[FieldElement<F>],
    transcript: &mut T,
) -> FieldElement<F>
where
    FieldElement<F>: ByteConversion,
{
    loop {
        let value: FieldElement<F> = transcript_to_field(transcript);
        if !lde_roots_of_unity_coset.iter().any(|x| x == &value)
            && !trace_roots_of_unity.iter().any(|x| x == &value)
        {
            return value;
        }
    }
}

pub fn batch_sample_challenges<F: IsFFTField, T: Transcript>(
    size: usize,
    transcript: &mut T,
) -> Vec<FieldElement<F>>
where
    FieldElement<F>: ByteConversion,
{
    (0..size).map(|_| transcript_to_field(transcript)).collect()
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        field::{
            element::FieldElement,
            fields::{
                fft_friendly::stark_252_prime_field::Stark252PrimeField,
                montgomery_backed_prime_fields::{IsModulus, U256PrimeField},
            },
        },
        unsigned_integer::element::U256,
    };

    use crate::starks::transcript::randomness_to_field;

    #[test]
    fn test_stark_prime_field_random_to_field_32() {
        #[rustfmt::skip]
        let mut randomness: [u8; 32] = [
            248, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 32, 
        ];

        type FE = FieldElement<Stark252PrimeField>;
        let field_element: FE = randomness_to_field(&mut randomness);
        let expected_fe = FE::from(32u64);
        assert_eq!(field_element, expected_fe)
    }

    #[test]
    fn test_stark_prime_field_random_to_fiel_repeated_f_and_zero() {
        #[rustfmt::skip]
        let mut randomness: [u8; 32] = [
            255, 0, 255, 0, 255, 0, 255, 0,
            255, 0, 255, 0, 255, 0, 255, 0, 
            255, 0, 255, 0, 255, 0, 255, 0, 
            255, 0, 255, 0, 255, 0, 255, 0,
        ];

        type FE = FieldElement<Stark252PrimeField>;

        // 251 bits should be used (252 of StarkField - 1) to avoid duplicates
        // This leaves a 7
        let expected_fe = FE::from_hex_unchecked(
            "\
            0700FF00FF00FF00\
            FF00FF00FF00FF00\
            FF00FF00FF00FF00\
            FF00FF00FF00FF00",
        );

        let field_element: FE = randomness_to_field(&mut randomness);

        assert_eq!(field_element, expected_fe)
    }

    #[test]
    fn test_241_bit_random_to_field() {
        #[derive(Clone, Debug)]
        pub struct TestModulus;
        impl IsModulus<U256> for TestModulus {
            const MODULUS: U256 = U256::from_hex_unchecked(
                "\
                0001000000000011\
                0000000000000000\
                0000000000000000\
                0000000000000001",
            );
        }

        pub type TestField = U256PrimeField<TestModulus>;

        #[rustfmt::skip]
        let mut randomness: [u8; 32] = [
            255, 255, 255, 1, 2, 3, 4, 5, 
            6, 7, 8, 1, 2, 3, 4, 5, 
            6, 7, 8, 1, 2, 3, 4, 5, 
            6, 7, 8, 1, 2, 3, 4, 5,
        ];

        type FE = FieldElement<TestField>;

        let expected_fe = FE::from_hex_unchecked(
            "\
            0000FF0102030405\
            0607080102030405\
            0607080102030405\
            0607080102030405",
        );

        let field_element: FE = randomness_to_field(&mut randomness);

        assert_eq!(field_element, expected_fe);
    }

    #[test]
    fn test_249_bit_random_to_field() {
        #[derive(Clone, Debug)]
        pub struct TestModulus;
        impl IsModulus<U256> for TestModulus {
            const MODULUS: U256 = U256::from_hex_unchecked(
                "\
                0200000000000011\
                0000000000000000\
                0000000000000000\
                0000000000000001",
            );
        }

        pub type TestField = U256PrimeField<TestModulus>;

        #[rustfmt::skip]
        let mut randomness: [u8; 32] = [
            255, 0, 255, 0, 255, 0, 255, 0,
            255, 0, 255, 0, 255, 0, 255, 0, 
            255, 0, 255, 0, 255, 0, 255, 0, 
            255, 0, 255, 0, 255, 0, 255, 0,
        ];

        let expected_fe = FE::from_hex_unchecked(
            "\
                0100FF00FF00FF00\
                FF00FF00FF00FF00\
                FF00FF00FF00FF00\
                FF00FF00FF00FF00",
        );

        type FE = FieldElement<TestField>;

        let field_element: FE = randomness_to_field(&mut randomness);

        assert_eq!(field_element, expected_fe)
    }
}
