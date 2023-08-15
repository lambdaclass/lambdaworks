use crate::hash::poseidon::parameters::{DefaultPoseidonParams, PermutationParameters};
use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_math::field::{element::FieldElement, traits::IsPrimeField};

#[derive(Clone)]
pub struct TreePoseidon<F: IsPrimeField> {
    poseidon: Poseidon<F>,
}

impl<F> Default for TreePoseidon<F>
where
    F: IsPrimeField,
{
    fn default() -> Self {
        let params = PermutationParameters::new_with(DefaultPoseidonParams::CairoStark252);
        let poseidon = Poseidon::new_with_params(params);

        Self { poseidon }
    }
}

impl<F> IsMerkleTreeBackend for TreePoseidon<F>
where
    F: IsPrimeField,
{
    type Node = FieldElement<F>;
    type Data = FieldElement<F>;

    fn hash_data(&self, input: &FieldElement<F>) -> FieldElement<F> {
        self.poseidon.hash_single(input)
    }

    fn hash_new_parent(&self, left: &FieldElement<F>, right: &FieldElement<F>) -> FieldElement<F> {
        self.poseidon.hash(left, right)
    }
}
