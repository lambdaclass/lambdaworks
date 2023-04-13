use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField, traits::IsField},
};

use crate::hash::traits::IsCryptoHash;

use super::{merkle::MerkleTree, proof::Proof};

/// Small field useful for starks, sometimes called min i goldilocks
/// Used in miden and winterfell
// This field shouldn't be defined inside the merkle tree module
pub type Ecgfp5 = U64PrimeField<0xFFFF_FFFF_0000_0001_u64>;
pub type Ecgfp5FE = FieldElement<Ecgfp5>;

pub type TestMerkleTreeEcgfp = MerkleTree<Ecgfp5>;

pub type TestMerkleTreeBls12381 = MerkleTree<BLS12381PrimeField>;

pub type TestProofEcgfp5 = Proof<Ecgfp5>;

#[derive(Debug, Clone)]

/// This hasher is for testing purposes
/// It adds the fields
/// Under no circunstance it can be used in production
pub struct TestHasher;

impl TestHasher {
    pub const fn new() -> TestHasher {
        TestHasher
    }
}

impl<F: IsField> IsCryptoHash<F> for TestHasher {
    fn hash_one(&self, input: FieldElement<F>) -> FieldElement<F> {
        &input + &input
    }

    fn hash_two(&self, left: FieldElement<F>, right: FieldElement<F>) -> FieldElement<F> {
        left + right
    }
}
