use lambdaworks_crypto::{hash::traits::IsCryptoHash, merkle_tree::proof::Proof};
use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsField, IsTwoAdicField},
};

use crate::{air::frame::Frame, fri::fri_decommit::FriDecommitment};

#[derive(Debug)]
pub struct DeepConsistencyCheck<F: IsTwoAdicField, H: IsCryptoHash<F>> {
    pub lde_trace_merkle_roots: Vec<FieldElement<F>>,
    pub lde_trace_merkle_proofs: Vec<Vec<Proof<F, H>>>,
    pub lde_trace_frame: Frame<F>,
    pub composition_poly_evaluations: Vec<FieldElement<F>>,
    pub deep_poly_evaluation: FieldElement<F>,
}

#[derive(Debug, Clone)]
pub struct StarkQueryProof<F: IsField> {
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub fri_decommitment: FriDecommitment<F>,
}

#[derive(Debug)]
pub struct StarkProof<F: IsTwoAdicField, H: IsCryptoHash<F>> {
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub trace_ood_frame_evaluations: Frame<F>,
    pub deep_consistency_check: DeepConsistencyCheck<F, H>,
    pub composition_poly_ood_evaluations: Vec<FieldElement<F>>,
    pub query_list: Vec<StarkQueryProof<F>>,
}
