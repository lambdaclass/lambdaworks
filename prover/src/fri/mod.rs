mod fri_commitment;
pub mod fri_decommit;
mod fri_functions;
mod fri_merkle_tree;

use std::borrow::Borrow;

use crate::fri::fri_commitment::{FriCommitment, FriCommitmentVec};
use crate::fri::fri_functions::next_fri_layer;
pub use crate::fri::fri_merkle_tree::FriMerkleTree;
pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
pub use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::unsigned_integer::element::U384;
pub use lambdaworks_math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::Polynomial,
};

pub type F = crate::air::polynomials::U384PrimeField;
pub type FE = crate::air::polynomials::U384FieldElement;

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
) -> FriCommitment<FE> {
    // Merkle tree:
    //     - ret_evaluation
    //     - root
    //     - hasher
    // Create a new merkle tree with evaluation_i
    let merkle_tree = FriMerkleTree::build(&evaluation_i);

    // append the root of the merkle tree to the transcript
    let root = merkle_tree.root.clone();
    let root_bytes = (*root.value()).to_bytes_be();
    transcript.append(&root_bytes);

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
    transcript: &mut Transcript,
) -> FriCommitmentVec<FE> {
    let mut fri_commitment_list = FriCommitmentVec::new();
    let evaluation_0 = p_0.evaluate_slice(domain_0);

    let merkle_tree = FriMerkleTree::build(&evaluation_0);

    // append the root of the merkle tree to the transcript
    let root = merkle_tree.root.clone();
    let root_bytes = (*root.value()).to_bytes_be();
    transcript.append(&root_bytes);

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

    let mut last_coef = last_poly.coefficients.get(0).unwrap();

    while degree > 0 {
        // sample beta:
        // let beta_bytes = transcript.challenge();
        // let beta = FE::from_bytes_be(&beta_bytes).unwrap();
        let beta = FE::new(U384::from("4"));

        let (p_i, domain_i, evaluation_i) = next_fri_layer(&last_poly, &last_domain, &beta);

        let commitment_i = fri_commitment(&p_i, &domain_i, &evaluation_i, transcript);

        // append root of merkle tree to transcript
        let tree = &commitment_i.merkle_tree;
        let root = tree.root.clone();
        let root_bytes = (*root.value()).to_bytes_be();
        transcript.append(&root_bytes);

        fri_commitment_list.push(commitment_i);
        degree = p_i.degree();

        last_poly = p_i.clone();
        last_coef = last_poly.coefficients.get(0).unwrap();
        last_domain = domain_i.clone();
    }

    // append last value of the polynomial to the trasncript
    let last_coef_bytes = (*last_coef.value()).to_bytes_be();
    transcript.append(&last_coef_bytes);

    fri_commitment_list
}
