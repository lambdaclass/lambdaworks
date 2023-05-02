use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};

use crate::{air::frame::Frame, fri::fri_decommit::FriDecommitment};

#[derive(Debug, Clone)]
pub struct DeepPolynomialOpenings<F: IsFFTField> {
    pub lde_composition_poly_even_proof: Proof<F>,
    pub lde_composition_poly_even_evaluation: FieldElement<F>,
    pub lde_composition_poly_odd_proof: Proof<F>,
    pub lde_composition_poly_odd_evaluation: FieldElement<F>,
    pub lde_trace_merkle_proofs: Vec<Proof<F>>,
    pub lde_trace_evaluations: Vec<FieldElement<F>>,
}

#[derive(Debug)]
pub struct StarkProof<F: IsFFTField> {
    // Commitments of the trace columns
    // [tⱼ]
    pub lde_trace_merkle_roots: Vec<FieldElement<F>>,
    // tⱼ(zgᵏ)
    pub trace_ood_frame_evaluations: Frame<F>,
    // [H₁]
    pub composition_poly_even_root: FieldElement<F>,
    // H₁(z²)
    pub composition_poly_even_ood_evaluation: FieldElement<F>,
    // [H₂]
    pub composition_poly_odd_root: FieldElement<F>,
    // H₂(z²)
    pub composition_poly_odd_ood_evaluation: FieldElement<F>,
    // [pₖ]
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    // pₙ
    pub fri_last_value: FieldElement<F>,
    // Open(p₀(D₀), 𝜐ₛ), Opwn(pₖ(Dₖ), −𝜐ₛ^(2ᵏ))
    pub query_list: Vec<FriDecommitment>,
    // Open(H₁(D_LDE, 𝜐₀), Open(H₂(D_LDE, 𝜐₀), Open(tⱼ(D_LDE), 𝜐₀)
    pub deep_poly_openings: DeepPolynomialOpenings<F>,
}
