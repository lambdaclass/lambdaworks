mod fri_commitment;
mod fri_decommit;
mod fri_functions;
mod fri_merkle_tree;

use crate::fri::fri_commitment::{FriCommitment, FriCommitmentVec};
use crate::fri::fri_functions::next_fri_layer;
use crate::fri::fri_merkle_tree::FriTestHasher;
pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::MerkleTree;
use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};
pub use lambdaworks_math::polynomial::Polynomial;

const ORDER: u64 = 293;
pub type F = U64PrimeField<ORDER>;
pub type FE = FieldElement<F>;

/// # Params
///
/// p_0,
/// original domain,
/// evaluation.
pub fn fri_commitment(
    p_i: &Polynomial<FieldElement<F>>,
    domain_i: &[FE],
    evaluation_i: &[FE],
    transcript: &mut Transcript,
) -> FriCommitment<F, FriTestHasher> {
    // Merkle tree:
    //     - ret_evaluation
    //     - root
    //     - hasher
    // Create a new merkle tree with evaluation_i
    let merkle_tree = MerkleTree::build(&evaluation_i, FriTestHasher);
    let root = merkle_tree.root.borrow().hash;
    // TODO @@@ let bytes = root.as_bytes();
    //transcript.append(bytes);

    FriCommitment {
        poly: p_i.clone(),
        domain: domain_i.to_vec(),
        evaluation: evaluation_i.to_vec(),
        merkle_tree,
    }
}

pub fn fri(
    p_0: &mut Polynomial<FieldElement<F>>,
    domain_0: &[FE],
) -> FriCommitmentVec<F, FriTestHasher> {
    let mut fri_commitment_list = FriCommitmentVec::new();
    let mut transcript = Transcript::new();

    let evaluation_0 = p_0.evaluate_slice(domain_0);
    let merkle_tree = MerkleTree::build(&evaluation_0, FriTestHasher);
    let commitment_0 = FriCommitment {
        poly: p_0.clone(),
        domain: domain_0.to_vec(),
        evaluation: evaluation_0.to_vec(),
        merkle_tree,
    };

    // last poly of the list
    let mut last_poly: Polynomial<FE> = p_0.clone();
    // last domain of the list
    let mut last_domain: Vec<FE> = domain_0.to_vec();

    // first evaluation in the list
    fri_commitment_list.push(commitment_0);
    let mut degree = p_0.degree();

    // TODO@@@ append root of the merkle tree to the transcript

    while degree > 0 {
        // sample beta:
        // TODO! let beta = transcript.challenge();
        let beta = FE::new(5);

        let (p_i, domain_i, evaluation_i) = next_fri_layer(&last_poly, &last_domain, &beta);

        let commitment_i = fri_commitment(&p_i, &domain_i, &evaluation_i, &mut transcript);

        fri_commitment_list.push(commitment_i);
        degree = p_i.degree();

        last_poly = p_i.clone();
        last_domain = domain_i.clone();

        // TODO
        // append root of merkle tree to transcript
    }

    // TODO
    // append last value of the polynomial to the trasncript

    fri_commitment_list
}
