use lambdaworks_math::traits::ByteConversion;
use miden_core::Felt;
use sha3::{Digest, Keccak256};
use stark_platinum_prover::{
    fri::FieldElement, prover::IsStarkProver, transcript::IsStarkTranscript,
    verifier::IsStarkVerifier,
};
use winter_math::StarkField;

pub mod air;
pub mod public_inputs;

pub struct Prover;
impl IsStarkProver for Prover {
    type Field = Felt;
}

pub struct Verifier {}
impl IsStarkVerifier for Verifier {
    type Field = Felt;
}

pub struct Transcript {
    hasher: Keccak256,
}

impl Transcript {
    pub fn new(data: &[u8]) -> Self {
        let mut res = Self {
            hasher: Keccak256::new(),
        };
        res.append_bytes(data);
        res
    }
}

impl IsStarkTranscript<Felt> for Transcript {
    fn append_field_element(&mut self, element: &FieldElement<Felt>) {
        self.append_bytes(&element.value().to_bytes_be());
    }

    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.hasher.update(&mut new_bytes.to_owned());
    }

    fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    fn sample_field_element(&mut self) -> FieldElement<Felt> {
        let mut bytes = self.state()[..8].try_into().unwrap();
        let mut x = u64::from_be_bytes(bytes);
        while x >= Felt::MODULUS {
            self.append_bytes(&bytes);
            bytes = self.state()[..8].try_into().unwrap();
            x = u64::from_be_bytes(bytes);
        }
        FieldElement::const_from_raw(Felt::new(x))
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        u64::from_be_bytes(self.state()[..8].try_into().unwrap()) % upper_bound
    }
}
