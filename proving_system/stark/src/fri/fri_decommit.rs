pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

#[derive(Debug, Clone)]
pub struct FriDecommitment {
    pub layers_auth_paths_sym: Vec<Proof<Stark252PrimeField>>,
    pub layers_evaluations_sym: Vec<FieldElement<Stark252PrimeField>>,
    pub first_layer_evaluation: FieldElement<Stark252PrimeField>,
    pub first_layer_auth_path: Proof<Stark252PrimeField>,
}
