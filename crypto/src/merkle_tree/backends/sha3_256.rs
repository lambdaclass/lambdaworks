use crate::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};
use sha3::{Digest, Sha3_256};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Sha3_256Tree<F> {
    phantom: PhantomData<F>,
}

impl<F> Default for Sha3_256Tree<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> IsMerkleTreeBackend for Sha3_256Tree<F>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    type Node = [u8; 32];
    type Data = FieldElement<F>;

    fn hash_data(&self, input: &FieldElement<F>) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(input.to_bytes_be());
        hasher.finalize().into()
    }

    fn hash_new_parent(&self, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{
        backends::sha3_256::Sha3_256Tree, merkle::MerkleTree, test_merkle::TestBackend,
    };

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;
    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn build_merkle_tree_from_an_odd_set_of_leaves() {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values);
        assert_eq!(merkle_tree.root, FE::new(8));
    }

    #[test]
    fn hash_data_field_element_backend_works() {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<Sha3_256Tree<U64PF>>::build(&values);
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<Sha3_256Tree<U64PF>>(&merkle_tree.root, 0, &values[0]));
    }
}
