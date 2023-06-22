
use std::marker::PhantomData;
use lambdaworks_math::{field::{traits::IsField, element::FieldElement}, traits::ByteConversion};
use sha3::{Digest, Sha3_512};
use crate::merkle_tree::traits::IsMerkleTreeBackend;

#[derive(Clone)]
pub struct Sha3_512Tree<F> {
    phantom: PhantomData<F>,
}

impl<F> Default for Sha3_512Tree<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> IsMerkleTreeBackend for Sha3_512Tree<F>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    type Node = [u8; 64];
    type Data = FieldElement<F>;

    fn hash_data(&self, input: &FieldElement<F>) -> [u8; 64] {
        let mut hasher = Sha3_512::new();
        hasher.update(input.to_bytes_be());
        hasher.finalize().into()
    }

    fn hash_new_parent(&self, left: &[u8; 64], right: &[u8; 64]) -> [u8; 64] {
        let mut hasher = Sha3_512::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }

    fn hash_leaves(&self, unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        unhashed_leaves
            .iter()
            .map(|leaf| self.hash_data(leaf))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{fields::u64_prime_field::U64PrimeField, element::FieldElement};

    use crate::merkle_tree::{merkle::MerkleTree, backends::{sha3_512::Sha3_512Tree}};

    
    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;
    #[test]
    fn hash_data_field_element_backend_works() {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<Sha3_512Tree<U64PF>>::build(&values);
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<Sha3_512Tree<U64PF>>(&merkle_tree.root, 0, &values[0]));
    }
}

