use super::FE;
use crate::{fri::fri_commitment::FriCommitmentVec, U384FieldElement, U384PrimeField};
pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::{hash::traits::IsCryptoHash, merkle_tree::DefaultHasher};

use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::{
    field::traits::IsField, traits::ByteConversion, unsigned_integer::element::U384,
};

#[derive(Debug, Clone)]
pub struct FriDecommitment {
    pub layer_merkle_paths: Vec<(
        Proof<U384PrimeField, DefaultHasher>,
        Proof<U384PrimeField, DefaultHasher>,
    )>,
    pub last_layer_evaluation: U384FieldElement,
}

// verifier chooses a randomness and get the index where
// they want to evaluate the poly
// TODO: encapsulate the return type of this function in a struct.
// This returns a list of authentication paths for evaluations on points and their symmetric counterparts.
pub fn fri_decommit_layers(
    commit: &FriCommitmentVec<FE>,
    index_to_verify: usize,
) -> FriDecommitment {
    let mut index = index_to_verify;

    let mut layer_merkle_paths = vec![];

    // with every element of the commit, we look for that one in
    // the merkle tree and get the corresponding element
    for commit_i in commit {
        let length_i = commit_i.domain.len();
        index = index % length_i;
        let evaluation_i = commit_i.evaluation[index].clone();
        let auth_path = commit_i.merkle_tree.get_proof(&evaluation_i).unwrap();

        // symmetrical element
        let index_sym = (index + length_i / 2) % length_i;
        let evaluation_i_sym = commit_i.evaluation[index_sym].clone();
        let auth_path_sym = commit_i.merkle_tree.get_proof(&evaluation_i_sym).unwrap();

        layer_merkle_paths.push((auth_path, auth_path_sym));
    }

    // send the last element of the polynomial
    let last = commit.last().unwrap();
    let last_evaluation = last.poly.coefficients[0].clone();

    return FriDecommitment {
        layer_merkle_paths: layer_merkle_paths,
        last_layer_evaluation: last_evaluation,
    };
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
    pub type FE_goldilocks = FieldElement<F>;

    #[test]
    fn test() {
        let subgroup_size = 1024_u64;
        let generator_field = FE_goldilocks::new(PRIME_GENERATOR.1);
        let exp = (PRIME_GENERATOR.0 - 1) / subgroup_size;
        let generator_of_subgroup = generator_field.pow(exp);
        let mut numbers = HashSet::new();

        let mut i = 0;
        for exp in 0..1024_u64 {
            i += 1;
            let ret = generator_of_subgroup.pow(exp);
            numbers.insert(ret.value().clone());
            println!("{ret:?}");
        }

        let count = numbers.len();
        println!("count: {count}");
        println!("iter: {i}");
    }
}
