pub mod fri_commitment;
pub mod fri_decommit;
mod fri_functions;
use crate::fri::fri_commitment::FriLayer;
use crate::transcript_to_field;
use lambdaworks_crypto::hash::sha3::Sha3Hasher;

pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
pub use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::ByteConversion;
pub use lambdaworks_math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::Polynomial,
};

use self::fri_functions::{fold_polynomial, next_domain};

pub type FriMerkleTree<F> = MerkleTree<F>;
pub(crate) const HASHER: Sha3Hasher = Sha3Hasher::new();

pub fn fri_commit_phase<F: IsField, T: Transcript>(
    number_layers: usize,
    p_0: Polynomial<FieldElement<F>>,
    domain_0: &[FieldElement<F>],
    transcript: &mut T,
) -> (FieldElement<F>, Vec<FriLayer<F>>)
where
    FieldElement<F>: ByteConversion,
{
    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut current_layer = FriLayer::new(p_0, domain_0);

    // TODO: last layer is unnecesary. We should remove it
    for _ in 0..number_layers {
        fri_layer_list.push(current_layer.clone());
        transcript.append(&current_layer.merkle_tree.root.to_bytes_be());
        let zeta = transcript_to_field(transcript);
        let next_poly = fold_polynomial(&current_layer.poly, &zeta);
        let next_domain = next_domain(&current_layer.domain);
        current_layer = FriLayer::new(next_poly, &next_domain);
    }
    fri_layer_list.push(current_layer.clone());

    let last_value = current_layer
        .poly
        .coefficients()
        .get(0)
        .unwrap_or(&FieldElement::zero())
        .clone();

    // append last value of the polynomial to the transcript
    let last_coef_bytes = last_value.to_bytes_be();
    transcript.append(&last_coef_bytes);

    (last_value, fri_layer_list)
}
