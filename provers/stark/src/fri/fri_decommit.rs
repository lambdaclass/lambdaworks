pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::proof::Proof;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsPrimeField;

use crate::config::Commitment;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FriDecommitment<F: IsPrimeField> {
    pub layers_auth_paths: Vec<Proof<Commitment>>,
    pub layers_evaluations_sym: Vec<FieldElement<F>>,
}
