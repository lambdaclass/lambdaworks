use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};

use super::HASHER;
pub use super::{FriMerkleTree, Polynomial};

#[derive(Clone)]
pub struct FriLayer<F: IsField> {
    pub poly: Polynomial<FieldElement<F>>,
    pub domain: Vec<FieldElement<F>>,
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: FriMerkleTree<F>,
}

impl<F> FriLayer<F>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    pub fn new(poly: Polynomial<FieldElement<F>>, domain: &[FieldElement<F>]) -> Self {
        let evaluation = poly.evaluate_slice(domain);
        let merkle_tree = FriMerkleTree::build(&evaluation, Box::new(HASHER));

        Self {
            poly,
            domain: domain.to_vec(),
            evaluation: evaluation.to_vec(),
            merkle_tree,
        }
    }
}
