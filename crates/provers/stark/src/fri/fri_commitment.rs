use lambdaworks_crypto::merkle_tree::{merkle::MerkleTree, traits::IsMerkleTreeBackend};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

use crate::multi_table_tree::MultiTableTree;

#[derive(Clone)]
pub struct FriLayer<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: MultiTableTree<F>,
    pub coset_offset: FieldElement<F>,
    pub domain_size: usize,
}

impl<F> FriLayer<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    pub fn new(
        evaluation: &[FieldElement<F>],
        merkle_tree: MultiTableTree<F>,
        coset_offset: FieldElement<F>,
        domain_size: usize,
    ) -> Self {
        Self {
            evaluation: evaluation.to_vec(),
            merkle_tree,
            coset_offset,
            domain_size,
        }
    }
}
