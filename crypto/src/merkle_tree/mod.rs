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

#[cfg(test)]

mod tests {

    use crate::merkle_tree::{proof::Proof, U64MerkleTree, U64Proof, U64FE};
    use lambdaworks_math::traits::ByteConversion;

    #[test]
    fn test() {
        let values: Vec<U64FE> = (1..6).map(U64FE::new).collect();
        let merkle_tree = U64MerkleTree::build(&values);
        let proof = merkle_tree.get_proof(&U64FE::new(2)).unwrap();
        let serialize_proof = proof.to_bytes_be();
        let proof: U64Proof = Proof::from_bytes_be(&serialize_proof).unwrap();
        assert!(U64MerkleTree::verify(&proof, merkle_tree.root));
    }

    #[test]
    fn create_a_merkle_tree_with_10000_elements_and_verify_that_an_element_is_part_of_it() {
        let values: Vec<U64FE> = (1..10000).map(U64FE::new).collect();
        let merkle_tree = U64MerkleTree::build(&values);
        let proof = merkle_tree.get_proof(&U64FE::new(9350)).unwrap();
        assert!(U64MerkleTree::verify(&proof, merkle_tree.root));
    }
}
