use crate::hash::poseidon::parameters::{DefaultPoseidonParams, PermutationParameters};
use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsPrimeField},
    traits::ByteConversion,
};

#[derive(Clone)]
pub struct BatchPoseidonTree<F: IsPrimeField> {
    poseidon: Poseidon<F>,
}

impl<F> Default for BatchPoseidonTree<F>
where
    F: IsPrimeField,
    FieldElement<F>: ByteConversion,
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
    FieldElement<F>: ByteConversion,
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

// #[cfg(test)]
// mod tests {
//     use lambdaworks_math::field::{
//         element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
//     };
//     use sha3::{Keccak256, Sha3_256};

//     use crate::merkle_tree::{backends::batch_256_bits::Batch256BitsTree, merkle::MerkleTree};

//     type F = Stark252PrimeField;
//     type FE = FieldElement<F>;

//     #[test]
//     fn hash_data_field_element_backend_works_with_sha3_256() {
//         let values = [
//             vec![FE::from(2u64), FE::from(11u64)],
//             vec![FE::from(3u64), FE::from(14u64)],
//             vec![FE::from(4u64), FE::from(7u64)],
//             vec![FE::from(5u64), FE::from(3u64)],
//             vec![FE::from(6u64), FE::from(5u64)],
//             vec![FE::from(7u64), FE::from(16u64)],
//             vec![FE::from(8u64), FE::from(19u64)],
//             vec![FE::from(9u64), FE::from(21u64)],
//         ];
//         let merkle_tree = MerkleTree::<Batch256BitsTree<F, Sha3_256>>::build(&values);
//         let proof = merkle_tree.get_proof_by_pos(0).unwrap();
//         assert!(proof.verify::<Batch256BitsTree<F, Sha3_256>>(&merkle_tree.root, 0, &values[0]));
//     }

//     #[test]
//     fn hash_data_field_element_backend_works_with_keccak256() {
//         let values = [
//             vec![FE::from(2u64), FE::from(11u64)],
//             vec![FE::from(3u64), FE::from(14u64)],
//             vec![FE::from(4u64), FE::from(7u64)],
//             vec![FE::from(5u64), FE::from(3u64)],
//             vec![FE::from(6u64), FE::from(5u64)],
//             vec![FE::from(7u64), FE::from(16u64)],
//             vec![FE::from(8u64), FE::from(19u64)],
//             vec![FE::from(9u64), FE::from(21u64)],
//         ];
//         let merkle_tree = MerkleTree::<Batch256BitsTree<F, Keccak256>>::build(&values);
//         let proof = merkle_tree.get_proof_by_pos(0).unwrap();
//         assert!(proof.verify::<Batch256BitsTree<F, Keccak256>>(&merkle_tree.root, 0, &values[0]));
//     }
// }
