pub mod air;
pub mod cairo_vm;
pub mod fri;
pub mod proof;
pub mod prover;
pub mod verifier;

use lambdaworks_crypto::{fiat_shamir::transcript::Transcript};
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    traits::IsField,
};

pub struct ProofConfig {
    pub count_queries: usize,
    pub blowup_factor: usize,
}

pub type PrimeField = Stark252PrimeField;
pub type FE = FieldElement<PrimeField>;

// TODO: change this to use more bits
pub fn transcript_to_field<F: IsField>(transcript: &mut Transcript) -> FieldElement<F> {
    let value: u64 = u64::from_be_bytes(transcript.challenge()[..8].try_into().unwrap());
    FieldElement::from(value)
}

pub fn transcript_to_usize(transcript: &mut Transcript) -> usize {
    const CANT_BYTES_USIZE: usize = (usize::BITS / 8) as usize;
    let value = transcript.challenge()[..CANT_BYTES_USIZE]
        .try_into()
        .unwrap();
    usize::from_be_bytes(value)
}

pub fn sample_z_ood<F: IsField>(
    lde_roots_of_unity_coset: &[FieldElement<F>],
    trace_roots_of_unity: &[FieldElement<F>],
    transcript: &mut Transcript,
) -> FieldElement<F> {
    loop {
        let value: FieldElement<F> = transcript_to_field(transcript);
        if !lde_roots_of_unity_coset.iter().any(|x| x == &value)
            && !trace_roots_of_unity.iter().any(|x| x == &value)
        {
            return value;
        }
    }
}
