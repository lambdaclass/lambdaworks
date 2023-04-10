use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField, traits::IsField},
};

use crate::hash::traits::IsCryptoHash;

use self::{merkle::MerkleTree, proof::Proof};

pub mod merkle;
pub mod proof;
mod utils;

/// Small field useful for starks
/// Used in miden and winterfell
pub type Ecgfp5 = U64PrimeField<0xFFFF_FFFF_0000_0001_u64>;
pub type Ecgfp5FE = FieldElement<Ecgfp5>;

pub type TestMerkleTreeEcgfp = MerkleTree<Ecgfp5, TestHasher>;

pub type TestMerkleTreeBls12381 = MerkleTree<BLS12381PrimeField, TestHasher>;

pub type TestProofEcgfp5 = Proof<Ecgfp5, TestHasher>;

#[derive(Debug, Clone)]

/// This hasher is for testing purposes
/// It adds the fields
/// Under no circunstance it can be used in production
pub struct TestHasher;

impl<F: IsField> IsCryptoHash<F> for TestHasher {
    fn new() -> TestHasher {
        TestHasher
    }

    fn hash_one(&self, input: FieldElement<F>) -> FieldElement<F> {
        &input + &input
    }

    fn hash_two(&self, left: FieldElement<F>, right: FieldElement<F>) -> FieldElement<F> {
        left + right
    }
}
