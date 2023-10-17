use super::air::{CairoAIR, PublicInputs};
use crate::air::MemorySegment;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use serde::{Deserialize, Serialize};
use stark_platinum_prover::proof::options::ProofOptions;
use stark_platinum_prover::proof::stark::StarkProof;
use stark_platinum_prover::verifier::verify;
use std::collections::HashMap;
use std::ops::Range;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub struct Stark252PrimeFieldProof(StarkProof<Stark252PrimeField>);

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct FE(FieldElement<Stark252PrimeField>);

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySegmentMap(HashMap<MemorySegment, Range<u64>>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMap(HashMap<FE, FE>);

#[wasm_bindgen]
pub fn verify_cairo_proof_wasm(
    proof: &Stark252PrimeFieldProof,
    pub_input_serialized: &[u8],
    proof_options: &ProofOptions,
) -> bool {
    let pub_input: PublicInputs = serde_cbor::from_slice(pub_input_serialized).unwrap();
    verify::<Stark252PrimeField, CairoAIR>(&proof.0, &pub_input, proof_options)
}

#[wasm_bindgen]
pub fn deserialize_proof_wasm(proof: &[u8]) -> Stark252PrimeFieldProof {
    let proof_inner: StarkProof<Stark252PrimeField> = serde_cbor::from_slice(proof).unwrap();
    Stark252PrimeFieldProof(proof_inner)
}

#[wasm_bindgen]
pub fn new_proof_options(
    blowup_factor: u8,
    fri_number_of_queries: usize,
    coset_offset: usize,
    grinding_factor: u8,
) -> ProofOptions {
    ProofOptions {
        blowup_factor,
        fri_number_of_queries,
        coset_offset: coset_offset as u64,
        grinding_factor,
    }
}
