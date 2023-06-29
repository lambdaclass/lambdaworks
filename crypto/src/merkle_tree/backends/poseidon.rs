use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};

use crate::{hash::poseidon::Poseidon, merkle_tree::traits::IsMerkleTreeBackend};
use std::fmt::Debug;

#[derive(Clone)]
pub struct TreePoseidon<M, const NUM_LIMBS: usize>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    hasher: Poseidon<MontgomeryBackendPrimeField<M, NUM_LIMBS>>,
}

impl<M, const NUM_LIMBS: usize> Default for TreePoseidon<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    fn default() -> Self {
        let hasher = Poseidon::default();
        TreePoseidon { hasher }
    }
}

impl<M, const NUM_LIMBS: usize> IsMerkleTreeBackend for TreePoseidon<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    type Node = FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>;
    type Data = FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>;

    fn hash_data(&self, input: &Self::Data) -> Self::Node {
        // return first element of the state (unwraps to be removed after trait changes to return Result<>)
        // This clone could be removed
        self.hasher
            .hash(&[input.clone()])
            .unwrap()
            .first()
            .unwrap()
            .clone()
    }

    fn hash_new_parent(&self, left: &Self::Data, right: &Self::Data) -> Self::Node {
        // return first element of the state (unwraps to be removed after trait changes to return Result<>)
        self.hasher
            .hash(&[left.clone(), right.clone()])
            .unwrap()
            .first()
            .unwrap()
            .clone()
    }
}

#[cfg(test)]
mod test {
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::{
        MontgomeryConfigStark252PrimeField, Stark252PrimeField,
    };

    use crate::merkle_tree::merkle::MerkleTree;

    use super::*;
    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn test_hash() {
        let poseidon_backend = TreePoseidon::default();

        let a = FE::one();
        let b = FE::zero();

        poseidon_backend.hash_new_parent(&a, &b);
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn build_poseidon_merkle_tree_from_an_odd_set_of_leaves() {
        type PoseidonStarkFieldTree = TreePoseidon<MontgomeryConfigStark252PrimeField, 4>;

        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree = MerkleTree::<PoseidonStarkFieldTree>::build(&values);
        assert_eq!(merkle_tree.root, FE::from(8));
    }
}
