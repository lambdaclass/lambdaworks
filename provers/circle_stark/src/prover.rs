use super::config::{BatchedMerkleTree, Commitment};
use lambdaworks_math::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};

pub struct FRIProof;

pub struct CommitmentData {
    pub(crate) trace_polys: Vec<FieldElement<Mersenne31Field>>,
    pub(crate) lde_trace_merkle_tree: BatchedMerkleTree<Mersenne31Field>,
    pub(crate) lde_trace_merkle_root: Commitment,
}

pub fn prove(trace: Vec<FieldElement<Mersenne31Field>>) -> CommitmentData {
    let trace_polys: Vec<FieldElement<Mersenne31Field>>;
    let lde_trace_merkle_tree;
    let lde_trace_merkle_root;

    CommitmentData {
        trace_polys,
        lde_trace_merkle_tree,
        lde_trace_merkle_root,
    }
}
