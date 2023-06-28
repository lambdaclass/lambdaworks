use lambdaworks_math::field::traits::IsField;

use crate::{
    hash::poseidon::{parameters::Parameters, Poseidon},
    merkle_tree::traits::IsMerkleTreeBackend,
};

#[derive(Clone)]
pub struct TreePoseidon<F>
where
    F: IsField,
{
    hasher: Poseidon<F>,
}

impl<F> Default for TreePoseidon<F>
where
    F: IsField,
{
    fn default() -> Self {
        let hasher = Poseidon::default();
        Self { hasher }
    }
}

impl<F> TreePoseidon<F>
where
    F: IsField,
{
    pub fn new_with_params(params: Parameters<F>) -> Self {
        let hasher = Poseidon::new_with_params(params);
        TreePoseidon { hasher }
    }
}

impl<F> IsMerkleTreeBackend for TreePoseidon<F>
where
    F: IsField,
{
    type Node = FieldElement<F>;
    type Data = FieldElement<F>;

    fn hash_data(&self, input: &Self::Data) -> Self::Node {
        let mut hasher = Poseidon::<F>::new();
        // return first element of the state (unwraps to be removed after trait changes to return Result<>)
        // This clone could be removed
        self.hash(&[input.clone()])
            .unwrap()
            .first()
            .unwrap()
            .clone()
    }

    fn hash_new_parent(
        &self,
        left: &FieldElement<BLS12381PrimeField>,
        right: &FieldElement<BLS12381PrimeField>,
    ) -> FieldElement<BLS12381PrimeField> {
        // return first element of the state (unwraps to be removed after trait changes to return Result<>)
        self.hash(&[left.clone(), right.clone()])
            .unwrap()
            .first()
            .unwrap()
            .clone()
    }
}
