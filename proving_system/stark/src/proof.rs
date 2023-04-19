use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::field::{element::FieldElement, traits::IsTwoAdicField};

use crate::{air::frame::Frame, fri::fri_decommit::FriDecommitment};

#[derive(Debug, Clone)]
pub struct DeepConsistencyCheck<F: IsTwoAdicField> {
    pub lde_trace_merkle_roots: Vec<FieldElement<F>>,
    pub lde_trace_merkle_proofs: Vec<Proof<F>>,
    pub lde_trace_evaluations: Vec<FieldElement<F>>,
    pub composition_poly_evaluations: Vec<FieldElement<F>>,
}

#[derive(Debug, Clone)]
pub struct StarkQueryProof<F: IsTwoAdicField> {
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub fri_decommitment: FriDecommitment<F>,
    pub deep_consistency_check: DeepConsistencyCheck<F>,
}

#[derive(Debug)]
pub struct StarkProof<F: IsTwoAdicField> {
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub trace_ood_frame_evaluations: Frame<F>,
    pub composition_poly_ood_evaluations: Vec<FieldElement<F>>,
    pub query_list: Vec<StarkQueryProof<F>>,
}
