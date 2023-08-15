use crate::hash::poseidon::parameters::{DefaultPoseidonParams, PermutationParameters};
use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_math::field::{element::FieldElement, traits::IsPrimeField};

#[derive(Clone)]
pub struct BatchPoseidonTree<F: IsPrimeField> {
    poseidon: Poseidon<F>,
}

impl<F> Default for BatchPoseidonTree<F>
where
    F: IsPrimeField,
{
    fn default() -> Self {
        let params = PermutationParameters::new_with(DefaultPoseidonParams::CairoStark252);
        let poseidon = Poseidon::new_with_params(params);

        Self { poseidon }
    }
}

impl<F> IsMerkleTreeBackend for BatchPoseidonTree<F>
where
    F: IsPrimeField,
{
    type Node = FieldElement<F>;
    type Data = Vec<FieldElement<F>>;

    fn hash_data(&self, input: &Vec<FieldElement<F>>) -> FieldElement<F> {
        self.poseidon.hash_many(input)
    }

    fn hash_new_parent(&self, left: &FieldElement<F>, right: &FieldElement<F>) -> FieldElement<F> {
        self.poseidon.hash(left, right)
    }
}
