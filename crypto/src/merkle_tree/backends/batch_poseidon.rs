use crate::{hash::poseidon::Poseidon, merkle_tree::traits::IsMerkleTreeBackend};
use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};
use std::fmt::Debug;

#[derive(Clone)]
pub struct BatchPoseidon<M, const NUM_LIMBS: usize>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    hasher: Poseidon<MontgomeryBackendPrimeField<M, NUM_LIMBS>>,
}

impl<M, const NUM_LIMBS: usize> Default for BatchPoseidon<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    fn default() -> Self {
        let hasher = Poseidon::default();
        BatchPoseidon { hasher }
    }
}

impl<M, const NUM_LIMBS: usize> IsMerkleTreeBackend for BatchPoseidon<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    type Node = FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>;
    type Data = Vec<FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>>;

    fn hash_data(&self, input: &Self::Data) -> Self::Node {
        let mut hasher = D::new();
        for element in input.iter() {
            hasher.update(element.to_bytes_be());
        }
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(&self, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}
