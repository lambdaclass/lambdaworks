use crate::hash::traits::IsCryptoHash;
use lambdaworks_math::{
    errors::ByteConversionError,
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};

/// Stores a merkle path to some leaf.
/// Internally, the necessary hashes are stored from root to leaf in the
/// `merkle_path` field, in such a way that, if the merkle tree is of height `n`, the
/// `i`-th element of `merkle_path` is the sibling node in the `n - 1 - i`-th check
/// when verifying.
#[derive(Debug, Clone)]
pub struct Proof<F: IsField, H: IsCryptoHash<F>> {
    pub merkle_path: Vec<FieldElement<F>>,
    pub hasher: H,
}

impl<F: IsField, H: IsCryptoHash<F>> Proof<F, H> {
    pub fn verify(
        &self,
        root_hash: &FieldElement<F>,
        mut index: usize,
        value: &FieldElement<F>,
    ) -> bool {
        let mut hashed_value = self.hasher.hash_one(value.clone());

        for sibling_node in self.merkle_path.iter().rev() {
            if index % 2 == 0 {
                hashed_value = self.hasher.hash_two(hashed_value, sibling_node.clone());
            } else {
                hashed_value = self.hasher.hash_two(sibling_node.clone(), hashed_value);
            }

            index >>= 1;
        }

        root_hash == &hashed_value
    }
}

impl<F, H> ByteConversion for Proof<F, H>
where
    F: IsField,
    H: IsCryptoHash<F>,
    FieldElement<F>: ByteConversion,
{
    /// Returns the byte representation of the element in big-endian order.
    fn to_bytes_be(&self) -> Vec<u8> {
        let mut buffer: Vec<u8> = Vec::new();

        for value in self.merkle_path.iter() {
            for val in value.to_bytes_be().iter() {
                buffer.push(*val);
            }
        }

        buffer.to_vec()
    }

    /// Returns the byte representation of the element in little-endian order.
    fn to_bytes_le(&self) -> Vec<u8> {
        let mut buffer: Vec<u8> = Vec::new();

        for value in self.merkle_path.iter() {
            for val in value.to_bytes_le().iter() {
                buffer.push(*val);
            }
        }

        buffer.to_vec()
    }

    /// Returns the element from its byte representation in big-endian order.
    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let mut merkle_path = Vec::new();

        for elem in bytes[0..].chunks(8) {
            let field = FieldElement::from_bytes_be(elem)?;
            merkle_path.push(field);
        }

        Ok(Proof {
            merkle_path,
            hasher: H::new(),
        })
    }

    /// Returns the element from its byte representation in little-endian order.
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let mut merkle_path = Vec::new();

        for elem in bytes[0..].chunks(8) {
            let field = FieldElement::from_bytes_le(elem)?;
            merkle_path.push(field);
        }

        Ok(Proof {
            merkle_path,
            hasher: H::new(),
        })
    }
}

#[cfg(test)]
mod tests {

    use lambdaworks_math::traits::ByteConversion;

    use crate::merkle_tree::{
        merkle::MerkleTree, proof::Proof, test_merkle::{Ecgfp5FE, TestProofEcgfp5, TestHasher, TestMerkleTreeEcgfp}
    };

    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    #[test]
    fn serialize_proof_and_deserialize_using_be_it_get_a_consistent_proof() {
        let merkle_path = [Ecgfp5FE::new(2), Ecgfp5FE::new(1), Ecgfp5FE::new(1)].to_vec();
        let original_proof = TestProofEcgfp5 {
            hasher: TestHasher,
            merkle_path,
        };
        let serialize_proof = original_proof.to_bytes_be();
        let proof: TestProofEcgfp5 = Proof::from_bytes_be(&serialize_proof).unwrap();

        for (o_node, node) in original_proof.merkle_path.iter().zip(proof.merkle_path) {
            assert_eq!(*o_node, node);
        }
    }

    #[test]
    fn serialize_proof_and_deserialize_using_le_it_get_a_consistent_proof() {
        let merkle_path = [Ecgfp5FE::new(2), Ecgfp5FE::new(1), Ecgfp5FE::new(1)].to_vec();
        let original_proof = TestProofEcgfp5 {
            hasher: TestHasher,
            merkle_path,
        };
        let serialize_proof = original_proof.to_bytes_le();
        let proof: TestProofEcgfp5 = Proof::from_bytes_le(&serialize_proof).unwrap();

        for (o_node, node) in original_proof.merkle_path.iter().zip(proof.merkle_path) {
            assert_eq!(*o_node, node);
        }
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree_when_given_the_leaf_position()
    {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&values);
        let proof = &merkle_tree.get_proof_by_pos(1).unwrap();
        assert_merkle_path(&proof.merkle_path, &[FE::new(2), FE::new(1), FE::new(1)]);
        assert!(proof.verify(&merkle_tree.root, 1, &FE::new(2)));
    }

    #[test]
    // expected | 2 | 1 | 1 |
    fn verify_a_proof_over_value_that_belongs_to_a_given_merkle_tree() {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&values);

        let proof = merkle_tree.get_proof(&FE::new(2)).unwrap();
        assert_merkle_path(&proof.merkle_path, &[FE::new(2), FE::new(1), FE::new(1)]);

        assert!(proof.verify(&merkle_tree.root, 1, &FE::new(2)));
    }

    #[test]
    fn merkle_proof_verifies_after_serialization_and_deserialization() {
        let values: Vec<Ecgfp5FE> = (1..6).map(Ecgfp5FE::new).collect();
        let merkle_tree = TestMerkleTreeEcgfp::build(&values);
        let proof = merkle_tree.get_proof(&Ecgfp5FE::new(2)).unwrap();
        let serialize_proof = proof.to_bytes_be();
        let proof: TestProofEcgfp5 = Proof::from_bytes_be(&serialize_proof).unwrap();
        assert!(proof.verify(&merkle_tree.root, 1, &Ecgfp5FE::new(2)));
    }

    #[test]
    fn create_a_merkle_tree_with_10000_elements_and_verify_that_an_element_is_part_of_it() {
        let values: Vec<Ecgfp5FE> = (1..10000).map(Ecgfp5FE::new).collect();
        let merkle_tree = TestMerkleTreeEcgfp::build(&values);
        let proof = merkle_tree.get_proof(&Ecgfp5FE::new(9350)).unwrap();
        assert!(proof.verify(&merkle_tree.root, 9349, &Ecgfp5FE::new(9350)));
    }

    fn assert_merkle_path(values: &[FE], expected_values: &[FE]) {
        for (node, expected_node) in values.iter().zip(expected_values) {
            assert_eq!(node, expected_node);
        }
    }
}
