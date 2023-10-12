use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};

use crate::{config::Commitment, frame::Frame, fri::fri_decommit::FriDecommitment};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeepPolynomialOpenings<F: IsFFTField> {
    pub lde_composition_poly_proof: Proof<Commitment>,
    pub lde_composition_poly_parts_evaluation: Vec<FieldElement<F>>,
    pub lde_trace_merkle_proofs: Vec<Proof<Commitment>>,
    pub lde_trace_evaluations: Vec<FieldElement<F>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct StarkProof<F: IsFFTField> {
    // Length of the execution trace
    pub trace_length: usize,
    // Commitments of the trace columns
    // [tⱼ]
    pub lde_trace_merkle_roots: Vec<Commitment>,
    // tⱼ(zgᵏ)
    pub trace_ood_frame_evaluations: Frame<F>,
    // Commitments to Hᵢ
    pub composition_poly_root: Commitment,
    // Hᵢ(z^N)
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<F>>,
    // [pₖ]
    pub fri_layers_merkle_roots: Vec<Commitment>,
    // pₙ
    pub fri_last_value: FieldElement<F>,
    // Open(p₀(D₀), 𝜐ₛ), Opwn(pₖ(Dₖ), −𝜐ₛ^(2ᵏ))
    pub query_list: Vec<FriDecommitment<F>>,
    // Open(H₁(D_LDE, 𝜐ᵢ), Open(H₂(D_LDE, 𝜐ᵢ), Open(tⱼ(D_LDE), 𝜐ᵢ)
    pub deep_poly_openings: Vec<DeepPolynomialOpenings<F>>,
    // Open(H₁(D_LDE, -𝜐ᵢ), Open(H₂(D_LDE, -𝜐ᵢ), Open(tⱼ(D_LDE), -𝜐ᵢ)
    pub deep_poly_openings_sym: Vec<DeepPolynomialOpenings<F>>,
    // nonce obtained from grinding
    pub nonce: u64,
}
