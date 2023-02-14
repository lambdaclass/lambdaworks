mod fri_commitment;
mod fri_decommit;
mod fri_functions;
mod fri_merkle_tree;

use crate::fri::fri_merkle_tree::FriTestHasher;
use fri_commitment::FriCommitment; //, FriCommitmentVec};
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

/*
pub fn fri(p0: &mut Polynomial<FieldElement<F>>, domain_0: &[FE])
-> FriCommitmentVec {
    let fri_commitment_list = FriCommitmentVec::new();
    let commitment_0 = FriCommitment {
        poly: p0.clone(),
        domain: Vec::from(domain_0),
        evaluation: Vec::new(), // calcular evaluation @@@@
        merkle_tree: String::new(), // TODO!
    };
    // TODO@@@ append root of the merkle tree to the transcript
    // first evaluation in the list
    fri_commitment_list.push(commitment_0);
    let mut degree = p0.degree();

    while degree > 0 {
        // sample beta:
        // beta = transcript.challenge();

        let (ret_poly, ret_next_domain, ret_evaluation) =
        next_fri_layer(
            poly: // last poly of the list
            domain: // last domain of the list
            beta: @@@@,
        );


        let commitment_i = fri_commitment(
            ret_poly,
            ret_next_domain,
            ret_evaluation,
            // merkle_tree0: ... TODO!!! from evaluation
        );

        degree = commitment_i.poly.degree();

        // TODO
        // append root of merkle tree to transcript

        fri_commitment_list.push(commitment_i);
    }

    // append last value of the polynomial to the trasncript

    fri_commitment_list
}
*/
