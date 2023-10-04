use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField},
    },
    traits::Serializable,
};

use crate::config::FriMerkleTree;

#[derive(Clone)]
pub struct FriLayer<F>
where
    F: IsField,
    FieldElement<F>: Serializable,
{
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: FriMerkleTree<F>,
    pub coset_offset: FieldElement<F>,
    pub domain_size: usize,
}

impl<F> FriLayer<F>
where
    F: IsField + IsFFTField,
    FieldElement<F>: Serializable,
{
    pub fn new(
        evaluation: &[FieldElement<F>],
        merkle_tree: FriMerkleTree<F>,
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
