use super::F;
use crate::fri::fri_commitment::FriCommitmentVec;
use crate::fri::fri_merkle_tree::FriTestHasher;
pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;

// verifier chooses a randomness and get the index where
// they want to evaluate the poly
pub fn fri_decommit_layers(
    commit: &FriCommitmentVec<F, FriTestHasher>,
    index_to_verify: usize,
    transcript: &mut Transcript,
) {
    let mut index = index_to_verify;
    // de cada componente del commit, voy al merkle tree y
    // me quedo con el elemento que corresponde
    for commit_i in commit {
        let length_i = commit_i.domain.len();
        index = index % length_i;
        let evaluation_i = commit_i.evaluation[index];
        let auth_path = commit_i.merkle_tree.get_proof(evaluation_i).unwrap();

        // symmetrical element
        let index_sym = (index + length_i / 2) % length_i;
        let evaluation_i_sym = commit_i.evaluation[index_sym];
        let auth_path_sym = commit_i.merkle_tree.get_proof(evaluation_i_sym).unwrap();

        // @@@ TODO! insert in transcript
    }

    // send the last element of the polynomial
    let last = commit.last().unwrap();
    let last_evaluation = last.poly.coefficients[0];

    // @@@ TODO insert last_evaluation in transcript


}

