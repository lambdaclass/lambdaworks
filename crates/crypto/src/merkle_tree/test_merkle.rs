use core::marker::PhantomData;

use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use super::{merkle::MerkleTree, traits::IsMerkleTreeBackend};

pub type TestMerkleTree<F> = MerkleTree<FieldElement<F>>;

#[derive(Debug, Clone)]

/// This hasher is for testing purposes
/// It adds the fields
/// Under no circunstance it can be used in production
pub struct TestBackend<F> {
    phantom: PhantomData<F>,
}

impl<F: IsField> Default for TestBackend<F> {
    fn default() -> Self {
        Self {
            phantom: Default::default(),
        }
    }
}

impl<F: IsField> IsMerkleTreeBackend for TestBackend<F>
where
    FieldElement<F>: Sync + Send,
{
    type Node = FieldElement<F>;
    type Data = FieldElement<F>;

    fn hash_data(input: &Self::Data) -> Self::Node {
        input + input
    }

    fn hash_new_parent(left: &Self::Node, right: &Self::Node) -> Self::Node {
        left + right
    }
}
