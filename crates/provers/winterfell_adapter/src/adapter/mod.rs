use lambdaworks_math::{field::fields::winterfell::QuadFelt, traits::ByteConversion};
use miden_core::Felt;
use sha3::{Digest, Keccak256};
use stark_platinum_prover::{fri::FieldElement, transcript::IsStarkTranscript};
use winter_math::StarkField;

pub mod air;
pub mod public_inputs;

pub struct FeltTranscript {
    hasher: Keccak256,
}

impl FeltTranscript {
    pub fn new(data: &[u8]) -> Self {
        let mut res = Self {
            hasher: Keccak256::new(),
        };
        res.append_bytes(data);
        res
    }

    fn sample(&mut self) -> [u8; 32] {
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&self.hasher.finalize_reset());
        result_hash.reverse();
        self.hasher.update(result_hash);
        result_hash
    }
}

impl IsStarkTranscript<Felt> for FeltTranscript {
    fn append_field_element(&mut self, element: &FieldElement<Felt>) {
        self.append_bytes(&element.value().to_bytes_be());
    }

    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.hasher.update(new_bytes);
    }

    fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    fn sample_field_element(&mut self) -> FieldElement<Felt> {
        loop {
            let bytes = self.sample();
            let candidate = u64::from_be_bytes(bytes[..8].try_into()
                .expect("sample() returns 32-byte array, slice of first 8 bytes always fits [u8; 8]"));
            if candidate < Felt::MODULUS {
                return FieldElement::const_from_raw(Felt::new(candidate));
            }
        }
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        assert!(upper_bound > 0, "upper_bound must be greater than 0");
        let zone = u64::MAX - (u64::MAX % upper_bound);
        loop {
            let bytes = self.sample();
            let candidate = u64::from_be_bytes(bytes[..8].try_into()
                .expect("sample() returns 32-byte array, slice of first 8 bytes always fits [u8; 8]"));
            if candidate < zone {
                return candidate % upper_bound;
            }
        }
    }
}

pub struct QuadFeltTranscript {
    felt_transcript: FeltTranscript,
}

impl QuadFeltTranscript {
    pub fn new(data: &[u8]) -> Self {
        Self {
            felt_transcript: FeltTranscript::new(data),
        }
    }
}

impl IsStarkTranscript<QuadFelt> for QuadFeltTranscript {
    fn append_field_element(&mut self, element: &FieldElement<QuadFelt>) {
        self.append_bytes(&element.value().to_bytes_be());
    }

    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.felt_transcript.append_bytes(new_bytes);
    }

    fn state(&self) -> [u8; 32] {
        self.felt_transcript.state()
    }

    fn sample_field_element(&mut self) -> FieldElement<QuadFelt> {
        let x = self.felt_transcript.sample_field_element();
        let y = self.felt_transcript.sample_field_element();
        FieldElement::const_from_raw(QuadFelt::new(*x.value(), *y.value()))
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        self.felt_transcript.sample_u64(upper_bound)
    }
}
