pub mod fri_commitment;
pub mod fri_decommit;
mod fri_functions;
use crate::fri::fri_commitment::FriLayer;
use crate::fri::fri_functions::next_fri_layer;
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

pub type FriMerkleTree<F> = MerkleTree<F>;
pub(crate) const HASHER: Sha3Hasher = Sha3Hasher::new();

/// # Params
///
/// p_0,
/// original domain,
/// evaluation.
pub fn fri_commitment<F: IsField>(
    p_i: &Polynomial<FieldElement<F>>,
    domain_i: &[FieldElement<F>],
    evaluation_i: &[FieldElement<F>],
) -> FriLayer<F>
where
    FieldElement<F>: ByteConversion,
{
    // Merkle tree:
    //     - ret_evaluation
    //     - root
    //     - hasher
    // Create a new merkle tree with evaluation_i
    let merkle_tree = FriMerkleTree::build(evaluation_i, Box::new(HASHER));

    FriLayer {
        poly: p_i.clone(),
        domain: domain_i.to_vec(),
        evaluation: evaluation_i.to_vec(),
        merkle_tree,
    }
}

pub fn fri_commit_phase<F: IsField, T: Transcript>(
    number_layers: usize,
    mut p_0: Polynomial<FieldElement<F>>,
    domain_0: &[FieldElement<F>],
    transcript: &mut T,
) -> Vec<FriLayer<F>>
where
    FieldElement<F>: ByteConversion,
{
    let mut fri_layer_list = Vec::new();
    let evaluation_0 = p_0.evaluate_slice(domain_0);

    let merkle_tree = FriMerkleTree::build(&evaluation_0, Box::new(HASHER));

    // append the root of the merkle tree to the transcript
    let root = merkle_tree.root.clone();
    transcript.append(&root.to_bytes_be());

    let commitment_0 = FriLayer {
        poly: p_0.clone(),
        domain: domain_0.to_vec(),
        evaluation: evaluation_0.to_vec(),
        merkle_tree,
    };

    // last poly of the list
    let mut last_poly: Polynomial<FieldElement<F>> = p_0.clone();
    // last domain of the list
    let mut last_domain: Vec<FieldElement<F>> = domain_0.to_vec();

    // first evaluation in the list
    fri_layer_list.push(commitment_0);

    let zero = FieldElement::zero();
    let mut last_coef = last_poly.coefficients.get(0).unwrap_or(&zero);

    for _ in 0..number_layers {
        let zeta = transcript_to_field(transcript);

        let layer_i = next_fri_layer(&last_poly, &last_domain, &zeta);

        // append root of merkle tree to transcript
        let root = layer_i.merkle_tree.root.clone();
        transcript.append(&root.to_bytes_be());

        last_poly = layer_i.poly.clone();
        last_coef = last_poly.coefficients.get(0).unwrap_or(&zero);

        last_domain = layer_i.domain.clone();

        fri_layer_list.push(layer_i);
    }

    // append last value of the polynomial to the trasncript
    let last_coef_bytes = last_coef.to_bytes_be();
    transcript.append(&last_coef_bytes);

    fri_layer_list
}
