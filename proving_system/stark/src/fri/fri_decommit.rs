pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

#[derive(Debug, Clone)]
pub struct FriDecommitment<F: IsField> {
    pub layers_auth_paths_sym: Vec<Proof<F>>,
    pub layers_evaluations_sym: Vec<FieldElement<F>>,
    pub first_layer_evaluation: FieldElement<F>,
    pub first_layer_auth_path: Proof<F>,
}

