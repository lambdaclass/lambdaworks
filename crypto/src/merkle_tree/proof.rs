use crate::hash::traits::IsCryptoHash;
use lambdaworks_math::{
    errors::ByteConversionError,
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        traits::IsField,
    },
    traits::ByteConversion,
    unsigned_integer::element::UnsignedInteger,
};

/// Stores a merkle path to some leaf.
/// Internally, the necessary hashes are stored from root to leaf in the
/// `merkle_path` field, in such a way that, if the merkle tree is of height `n`, the
/// `i`-th element of `merkle_path` is the sibling node in the `n - 1 - i`-th check
/// when verifying.
#[derive(Debug, Clone)]
pub struct Proof<F: IsField> {
    pub merkle_path: Vec<FieldElement<F>>,
}

impl<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize>
    Proof<MontgomeryBackendPrimeField<M, N>>
{
    pub fn verify(
        &self,
        root_hash: &FieldElement<MontgomeryBackendPrimeField<M, N>>,
        mut index: usize,
        value: &FieldElement<MontgomeryBackendPrimeField<M, N>>,
        hasher: &dyn IsCryptoHash<MontgomeryBackendPrimeField<M, N>>,
    ) -> bool {
        let mut hashed_value = hasher.hash_one(value);

        for sibling_node in self.merkle_path.iter() {
            if index % 2 == 0 {
                hashed_value = hasher.hash_two(&hashed_value, sibling_node);
            } else {
                hashed_value = hasher.hash_two(sibling_node, &hashed_value);
            }

            index >>= 1;
        }

        root_hash == &hashed_value
    }
}

impl<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize> ByteConversion
    for Proof<MontgomeryBackendPrimeField<M, N>>
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

        for elem in bytes[0..].chunks(N * 8) {
            let field = FieldElement::from_bytes_be(elem)?;
            merkle_path.push(field);
        }

        Ok(Proof { merkle_path })
    }

    /// Returns the element from its byte representation in little-endian order.
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let mut merkle_path = Vec::new();

        for elem in bytes[0..].chunks(N * 8) {
            let field = FieldElement::from_bytes_le(elem)?;
            merkle_path.push(field);
        }

        Ok(Proof { merkle_path })
    }
}

#[cfg(test)]
mod tests {

    use lambdaworks_math::{
        field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::ByteConversion,
    };

    use crate::merkle_tree::{merkle::MerkleTree, proof::Proof, test_merkle::TestHasher};

    use lambdaworks_math::field::element::FieldElement;

    type FE = FieldElement<Stark252PrimeField>;

    #[test]
    fn serialize_proof_and_deserialize_using_be_it_get_a_consistent_proof() {
        let merkle_path = [FE::from(2), FE::from(1), FE::from(1)].to_vec();
        let original_proof = Proof { merkle_path };
        let serialize_proof = original_proof.to_bytes_be();
        let proof = Proof::from_bytes_be(&serialize_proof).unwrap();

        for (o_node, node) in original_proof.merkle_path.iter().zip(proof.merkle_path) {
            assert_eq!(*o_node, node);
        }
    }

    #[test]
    fn serialize_proof_and_deserialize_using_le_it_get_a_consistent_proof() {
        let merkle_path = [FE::from(2), FE::from(1), FE::from(1)].to_vec();
        let original_proof = Proof { merkle_path };
        let serialize_proof = original_proof.to_bytes_le();
        let proof = Proof::from_bytes_le(&serialize_proof).unwrap();

        for (o_node, node) in original_proof.merkle_path.iter().zip(proof.merkle_path) {
            assert_eq!(*o_node, node);
        }
    }

    // Ignore this until we have a way to make a test like this for Stark252PrimeField or we
    // have MerkleTree for mini goldilocks
    #[ignore]
    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree_when_given_the_leaf_position()
    {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree = MerkleTree::build(&values, Box::new(TestHasher::new()));
        let proof = &merkle_tree.get_proof_by_pos(1).unwrap();
        assert_merkle_path(&proof.merkle_path, &[FE::from(2), FE::from(1), FE::from(1)]);
        assert!(proof.verify(&merkle_tree.root, 1, &FE::from(2), &TestHasher::new()));
    }

    #[test]
    fn merkle_proof_verifies_after_serialization_and_deserialization() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree = MerkleTree::build(&values, Box::new(TestHasher::new()));
        let proof = merkle_tree.get_proof_by_pos(1).unwrap();
        let serialize_proof = proof.to_bytes_be();
        let proof = Proof::from_bytes_be(&serialize_proof).unwrap();
        assert!(proof.verify(&merkle_tree.root, 1, &FE::from(2), &TestHasher::new()));
    }

    #[test]
    fn create_a_merkle_tree_with_10000_elements_and_verify_that_an_element_is_part_of_it() {
        let values: Vec<FE> = (1..10000).map(FE::from).collect();
        let merkle_tree = MerkleTree::build(&values, Box::new(TestHasher::new()));
        let proof = merkle_tree.get_proof_by_pos(9349).unwrap();
        assert!(proof.verify(&merkle_tree.root, 9349, &FE::from(9350), &TestHasher::new()));
    }

    fn assert_merkle_path(values: &[FE], expected_values: &[FE]) {
        for (node, expected_node) in values.iter().zip(expected_values) {
            assert_eq!(node, expected_node);
        }
    }
}
