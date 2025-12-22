use crate::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};

use core::marker::PhantomData;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField, IsSubFieldOf},
    },
    traits::ByteConversion,
};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use sha3::{Digest, Keccak256};

pub struct DefaultTranscript<F: HasDefaultTranscript> {
    hasher: Keccak256,
    phantom: PhantomData<F>,
}

impl<F> DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    pub fn new(data: &[u8]) -> Self {
        let mut res = Self {
            hasher: Keccak256::new(),
            phantom: PhantomData,
        };
        res.append_bytes(data);
        res
    }

    pub fn sample(&mut self) -> [u8; 32] {
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&self.hasher.finalize_reset());
        result_hash.reverse();
        self.hasher.update(result_hash);
        result_hash
    }
}

impl<F> Default for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    fn default() -> Self {
        Self::new(&[])
    }
}

impl<F> IsTranscript<F> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.hasher.update(new_bytes);
    }

    fn append_field_element(&mut self, element: &FieldElement<F>) {
        self.append_bytes(&element.to_bytes_be());
    }

    fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    fn sample_field_element(&mut self) -> FieldElement<F> {
        let mut rng = <ChaCha20Rng as SeedableRng>::from_seed(self.sample());
        F::get_random_field_element_from_rng(&mut rng)
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        u64::from_be_bytes(self.state()[..8].try_into().unwrap()) % upper_bound
    }
}

impl<F, S> IsStarkTranscript<F, S> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    S: IsField + IsSubFieldOf<F>,
{
    // nothing to implement: sample_z_ood uses the default body
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec::Vec;
    use lambdaworks_math::{
        elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField,
        field::fields::fft_friendly::{
            babybear_u32::Babybear31PrimeField,
            quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
        },
    };

    #[test]
    fn basic_challenge() {
        let mut transcript = DefaultTranscript::<FrField>::default();

        let point_a: Vec<u8> = vec![0xFF, 0xAB];
        let point_b: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

        transcript.append_bytes(&point_a); // point_a
        transcript.append_bytes(&point_b); // point_a + point_b

        let challenge1 = transcript.sample(); // Hash(point_a  + point_b)

        assert_eq!(
            challenge1,
            [
                0x0c, 0x2b, 0xd8, 0xcf, 0x2d, 0x71, 0xe0, 0x0a, 0xce, 0xa3, 0xbd, 0x5d, 0xc7, 0x9f,
                0x4f, 0x93, 0xed, 0x57, 0x42, 0xd0, 0x23, 0xbd, 0x47, 0xc9, 0x04, 0xc2, 0x67, 0x9d,
                0xbc, 0xfa, 0x7c, 0xa7
            ]
        );

        let point_c: Vec<u8> = vec![0xFF, 0xAB];
        let point_d: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

        transcript.append_bytes(&point_c); // Hash(point_a  + point_b) + point_c
        transcript.append_bytes(&point_d); // Hash(point_a  + point_b) + point_c + point_d

        let challenge2 = transcript.sample(); // Hash(Hash(point_a  + point_b) + point_c + point_d)
        assert_eq!(
            challenge2,
            [
                0x81, 0x61, 0x51, 0xc5, 0x7e, 0xcb, 0x45, 0xd5, 0x17, 0x1a, 0x3c, 0x2e, 0x38, 0x04,
                0x5d, 0xfb, 0x3a, 0x3d, 0x33, 0x8a, 0x22, 0xaf, 0xf8, 0x60, 0x85, 0xb9, 0x54, 0x3f,
                0xf8, 0x32, 0x32, 0xbc
            ]
        );
    }

    type FE = FieldElement<Babybear31PrimeField>;
    type Fp4E = FieldElement<Degree4BabyBearU32ExtensionField>;

    #[test]
    fn quartic_baby_bear_transcript_distinguish_different_fe() {
        let mut transcript_1 = DefaultTranscript::<Degree4BabyBearU32ExtensionField>::default();
        transcript_1.append_field_element(&Fp4E::new([
            FE::one(),
            FE::zero(),
            FE::zero(),
            FE::zero(),
        ]));
        let sample_1 = transcript_1.sample_field_element();

        let mut transcript_2 = DefaultTranscript::<Degree4BabyBearU32ExtensionField>::default();
        transcript_2.append_field_element(&Fp4E::new([
            FE::zero(),
            FE::zero(),
            FE::zero(),
            FE::one(),
        ]));
        let sample_2 = transcript_2.sample_field_element();

        let mut transcript_3 = DefaultTranscript::<Degree4BabyBearU32ExtensionField>::default();
        transcript_3.append_field_element(&Fp4E::new([
            FE::one(),
            FE::zero(),
            FE::zero(),
            FE::zero(),
        ]));
        let sample_3 = transcript_3.sample_field_element();

        assert!(sample_1 != sample_2);
        assert!(sample_1 == sample_3);
    }
}
