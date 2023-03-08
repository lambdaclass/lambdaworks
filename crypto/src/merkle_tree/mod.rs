use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField, traits::IsField},
};

use crate::hash::traits::IsCryptoHash;

use self::{merkle::MerkleTree, proof::Proof};

pub mod merkle;
pub mod proof;
mod utils;

pub type U64F = U64PrimeField<0xFFFF_FFFF_0000_0001_u64>;
pub type U64FE = FieldElement<U64F>;

pub type U64MerkleTree = MerkleTree<U64F, DefaultHasher>;
pub type MerkleTreeDefault = MerkleTree<BLS12381PrimeField, DefaultHasher>;

pub type U64Proof = Proof<U64F, DefaultHasher>;

#[derive(Debug, Clone)]
pub struct DefaultHasher;

impl<F: IsField> IsCryptoHash<F> for DefaultHasher {
    fn new() -> DefaultHasher {
        DefaultHasher
    }

    fn hash_one(&self, input: FieldElement<F>) -> FieldElement<F> {
        &input + &input
    }

    fn hash_two(&self, left: FieldElement<F>, right: FieldElement<F>) -> FieldElement<F> {
        left + right
    }
}
