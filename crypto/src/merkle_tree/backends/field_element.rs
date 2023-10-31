use crate::hash::poseidon::starknet::parameters::{DefaultPoseidonParams, PermutationParameters};
use crate::hash::poseidon::starknet::Poseidon;

use crate::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsField, IsPrimeField},
    },
    traits::Serializable,
};
use sha3::{
    digest::{generic_array::GenericArray, OutputSizeUser},
    Digest,
};
use std::marker::PhantomData;
#[derive(Clone)]
pub struct FieldElementBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default for FieldElementBackend<F, D, NUM_BYTES> {
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for FieldElementBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: Serializable,
    [u8; NUM_BYTES]: From<GenericArray<u8, <D as OutputSizeUser>::OutputSize>>,
{
    type Node = [u8; NUM_BYTES];
    type Data = FieldElement<F>;

    fn hash_data(&self, input: &FieldElement<F>) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(input.serialize());
        hasher.finalize().into()
    }

    fn hash_new_parent(&self, left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }
}

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

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };
    use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

    use crate::merkle_tree::{backends::field_element::FieldElementBackend, merkle::MerkleTree};

    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn hash_data_field_element_backend_works_with_keccak_256() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree = MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values);
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementBackend<F, Keccak256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_256() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree = MerkleTree::<FieldElementBackend<F, Sha3_256, 32>>::build(&values);
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementBackend<F, Sha3_256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_keccak_512() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree = MerkleTree::<FieldElementBackend<F, Keccak512, 64>>::build(&values);
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementBackend<F, Keccak512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_512() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree = MerkleTree::<FieldElementBackend<F, Sha3_512, 64>>::build(&values);
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementBackend<F, Sha3_512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }
}
