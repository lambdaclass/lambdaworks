pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::ByteConversion;

use super::fri_commitment::FriLayer;
#[derive(Debug, Clone)]
pub struct FriDecommitment<F: IsField> {
    pub layer_merkle_paths: Vec<(Proof<F>, Proof<F>)>,
    pub layer_evaluations: Vec<(FieldElement<F>, FieldElement<F>)>,
    pub last_layer_evaluation: FieldElement<F>,
}

// verifier chooses a randomness and get the index where
// they want to evaluate the poly
// TODO: encapsulate the return type of this function in a struct.
// This returns a list of authentication paths for evaluations on points and their symmetric counterparts.
pub fn fri_decommit_layers<F: IsField>(
    commit: &Vec<FriLayer<F>>,
    index_to_verify: usize,
) -> FriDecommitment<F>
where
    FieldElement<F>: ByteConversion,
{
    let mut index = index_to_verify;

    let mut layer_merkle_paths = vec![];
    let mut layer_evaluations = vec![];

    // with every element of the commit, we look for that one in
    // the merkle tree and get the corresponding element
    for commit_i in commit {
        let length_i = commit_i.domain.len();
        index %= length_i;
        let evaluation_i = commit_i.evaluation[index].clone();
        let auth_path = commit_i.merkle_tree.get_proof_by_pos(index).unwrap();

        // symmetrical element
        let index_sym = (index + length_i / 2) % length_i;
        let evaluation_i_sym = commit_i.evaluation[index_sym].clone();
        let auth_path_sym = commit_i.merkle_tree.get_proof_by_pos(index_sym).unwrap();

        layer_merkle_paths.push((auth_path, auth_path_sym));
        layer_evaluations.push((evaluation_i, evaluation_i_sym));
    }

    // send the last element of the polynomial
    let last = commit.last().unwrap();

    // This get can't fail, since the last will always have at least one evaluation
    // (in fact two, but on the last step the two will be the same because the polynomial will have
    // degree 0).
    let last_evaluation = last.evaluation[0].clone();

    FriDecommitment {
        layer_merkle_paths,
        layer_evaluations,
        last_layer_evaluation: last_evaluation,
    }
}

// Integration test:
// * get an arbitrary polynomial
// * have a domain containing roots of the unity (# is power of two)
// p = 65_537
// * apply FRI commitment
// * apply FRI decommitment
// assert:
// * evaluations of the polynomials coincide with calculations from the decommitment
// * show a fail example: with a monomial

#[cfg(test)]
mod tests {
    use crate::fri::U64PrimeField;
    use lambdaworks_math::field::element::FieldElement;
    use std::collections::HashSet;
    const PRIME_GENERATOR: (u64, u64) = (0xFFFF_FFFF_0000_0001_u64, 2717_u64);
    pub type F = U64PrimeField<{ PRIME_GENERATOR.0 }>;
    pub type FeGoldilocks = FieldElement<F>;

    #[test]
    fn test() {
        let subgroup_size = 1024_u64;
        let generator_field = FeGoldilocks::new(PRIME_GENERATOR.1);
        let exp = (PRIME_GENERATOR.0 - 1) / subgroup_size;
        let generator_of_subgroup = generator_field.pow(exp);
        let mut numbers = HashSet::new();

        let mut i = 0;
        for exp in 0..1024_u64 {
            i += 1;
            let ret = generator_of_subgroup.pow(exp);
            numbers.insert(*ret.value());
            println!("{ret:?}");
        }

        let count = numbers.len();
        println!("count: {count}");
        println!("iter: {i}");
    }
}
