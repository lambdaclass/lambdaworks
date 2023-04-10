use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsField, IsTwoAdicField},
};

use crate::{air::frame::Frame, fri::fri_decommit::FriDecommitment};

#[derive(Debug, Clone)]
pub struct StarkQueryProof<F: IsField> {
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub fri_decommitment: FriDecommitment<F>,
}

#[derive(Debug)]
pub struct StarkProof<F: IsTwoAdicField> {
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub trace_ood_frame_evaluations: Frame<F>,
    pub composition_poly_evaluations: Vec<FieldElement<F>>,
    pub query_list: Vec<StarkQueryProof<F>>,
}
