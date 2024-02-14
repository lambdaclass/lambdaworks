use lambdaworks_crypto::merkle_tree::proof::Proof;
use stark_platinum_prover::config::Commitment;

use crate::Felt252;

#[derive(Debug, Clone)]
pub struct StarkWitness {
    pub traces_decommitment: TracesDecommitment,
    pub traces_witness: TracesWitness,
    pub composition_decommitment: TableDecommitment,
    pub composition_witness: TableCommitmentWitness,
    pub fri_witness: FriWitness,
}

#[derive(Debug, Clone)]
pub struct TracesDecommitment {
    pub original: TableDecommitment,
    pub interaction: TableDecommitment,
}

#[derive(Debug, Clone)]
pub struct TableDecommitment {
    pub n_values: Felt252,
    pub values: Vec<Felt252>,
}

#[derive(Debug, Clone)]
pub struct TracesWitness {
    pub original: TableCommitmentWitness,
    pub interaction: TableCommitmentWitness,
}

#[derive(Debug, Clone)]
pub struct TableCommitmentWitness {
    pub vector: VectorCommitmentWitness,
}

#[derive(Debug, Clone)]
pub struct VectorCommitmentWitness {
    pub n_authentications: Felt252,
    pub authentications: Proof<Commitment>,
}

#[derive(Debug, Clone)]
pub struct FriWitness {
    pub layers: Vec<FriLayerWitness>,
}

#[derive(Debug, Clone)]
pub struct FriLayerWitness {
    pub n_leaves: Felt252,
    pub leaves: Vec<Felt252>,
    pub table_witness: TableCommitmentWitness,
}
