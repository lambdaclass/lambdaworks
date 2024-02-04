use std::marker::PhantomData;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::{
    field::{element::*, traits::IsField},
    traits::ByteConversion,
    unsigned_integer::element::U256,
};

pub struct PlonkTranscript<F: IsField> {
    state: [u8; 32],
    seed_increment: U256,
    counter: u32,
    spare_bytes: Vec<u8>,
    phantom: PhantomData<F>,
}

impl<F: IsField> PlonkTranscript<F> {
    pub fn new() -> Self {
        Self {
            state: Self::keccak_hash(&[]),
            seed_increment: U256::from_hex_unchecked("1"),
            counter: 0,
            spare_bytes: vec![],
            phantom: PhantomData,
        }
    }

    pub fn sample_block(&mut self, used_bytes: usize) -> Vec<u8> {
        let mut first_part: Vec<u8> = self.state.to_vec();
        let mut counter_bytes: Vec<u8> = vec![0; 28]
            .into_iter()
            .chain(self.counter.to_be_bytes().to_vec())
            .collect();
        self.counter += 1;
        first_part.append(&mut counter_bytes);
        let block = Self::keccak_hash(&first_part);
        self.spare_bytes.extend(&block[used_bytes..]);
        block[..used_bytes].to_vec()
    }

    pub fn sample(&mut self, num_bytes: usize) -> Vec<u8> {
        let num_blocks = num_bytes / 32;
        let mut result: Vec<u8> = Vec::new();

        for _ in 0..num_blocks {
            let mut block = self.sample_block(32);
            result.append(&mut block);
        }

        let rest = num_bytes % 32;
        if rest <= self.spare_bytes.len() {
            result.append(&mut self.spare_bytes[..rest].to_vec());
            self.spare_bytes.drain(..rest);
        } else {
            let mut block = self.sample_block(rest);
            result.append(&mut block);
        }

        result
    }
}

impl<F> Default for PlonkTranscript<F>
where
    F: IsField,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IsField> IsTranscript<F> for PlonkTranscript<F> {
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&self.state);
        result_hash.reverse();

        let digest = U256::from_bytes_be(&self.state).unwrap();
        let new_seed = (digest + self.seed_increment).to_bytes_be();
        self.state = Self::keccak_hash(&[&new_seed, new_bytes].concat());
        self.counter = 0;
        self.spare_bytes.clear();
    }

    fn append_field_element(&mut self, element: &FieldElement<F>) {
        // Self::append_bytes(self, &element.value().to_bytes_be());
        todo!()
    }

    fn state(&self) -> [u8; 32] {
        self.state
    }

    fn sample_field_element(&mut self) -> FieldElement<F> {
        todo!()
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate alloc;
    use alloc::vec::Vec;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField;

    #[test]
    fn basic_challenge() {
        let mut transcript = PlonkTranscript::<FrField>::new();

        let point_a: Vec<u8> = vec![0xFF, 0xAB];
        let point_b: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

        transcript.append_bytes(&point_a); // point_a
        transcript.append_bytes(&point_b); // point_a + point_b

        let challenge1 = transcript.sample(32); // Hash(point_a  + point_b)

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

        let challenge2 = transcript.sample(32); // Hash(Hash(point_a  + point_b) + point_c + point_d)
        assert_eq!(
            challenge2,
            [
                0x81, 0x61, 0x51, 0xc5, 0x7e, 0xcb, 0x45, 0xd5, 0x17, 0x1a, 0x3c, 0x2e, 0x38, 0x04,
                0x5d, 0xfb, 0x3a, 0x3d, 0x33, 0x8a, 0x22, 0xaf, 0xf8, 0x60, 0x85, 0xb9, 0x54, 0x3f,
                0xf8, 0x32, 0x32, 0xbc
            ]
        );
    }
}
