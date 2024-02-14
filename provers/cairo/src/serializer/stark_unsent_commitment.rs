use std::io;

use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark_platinum_prover::{config::Commitment, proof::stark::StarkProof, Felt252};

#[derive(Debug, Clone)]
pub struct StarkUnsentCommitment {
    pub traces: TracesUnsentCommitment,
    pub composition: Commitment,
    pub oods_values: Vec<Felt252>,
    pub fri: FriUnsentCommitment,
    pub proof_of_work: ProofOfWorkUnsentCommitment,
}

#[derive(Debug, Clone)]
pub struct TracesUnsentCommitment {
    pub original: Commitment,
    pub interaction: Commitment,
}

#[derive(Debug, Clone)]
pub struct FriUnsentCommitment {
    pub inner_layers: Vec<Commitment>,
    pub last_layer_coefficients: Vec<Felt252>,
}

#[derive(Debug, Clone)]
pub struct ProofOfWorkUnsentCommitment {
    pub nonce: u64,
}

impl StarkUnsentCommitment {
    pub fn convert(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
    ) -> Result<Self, io::Error> {
        let mut oods_values = vec![];

        for i in 0..proof.trace_ood_evaluations.width {
            for j in 0..proof.trace_ood_evaluations.height {
                oods_values.push(proof.trace_ood_evaluations.get_row(j)[i]);
            }
        }

        for elem in proof.composition_poly_parts_ood_evaluation.iter() {
            oods_values.push(*elem);
        }

        Ok(StarkUnsentCommitment {
            traces: TracesUnsentCommitment::convert(proof)?,
            composition: proof.composition_poly_root,
            oods_values,
            fri: FriUnsentCommitment::convert(proof),
            proof_of_work: ProofOfWorkUnsentCommitment::convert(proof)?,
        })
    }
}

impl TracesUnsentCommitment {
    pub fn convert(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
    ) -> Result<Self, io::Error> {
        Ok(TracesUnsentCommitment {
            original: proof.lde_trace_main_merkle_root,
            interaction: proof
                .lde_trace_aux_merkle_root
                .ok_or(io::Error::from(io::ErrorKind::InvalidData))?,
        })
    }
}

impl FriUnsentCommitment {
    pub fn convert(proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>) -> Self {
        FriUnsentCommitment {
            inner_layers: proof.fri_layers_merkle_roots.to_owned(),
            last_layer_coefficients: vec![proof.fri_last_value],
        }
    }
}

impl ProofOfWorkUnsentCommitment {
    pub fn convert(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
    ) -> Result<Self, io::Error> {
        Ok(ProofOfWorkUnsentCommitment {
            nonce: proof
                .nonce
                .ok_or(io::Error::from(io::ErrorKind::InvalidData))?,
        })
    }
}
