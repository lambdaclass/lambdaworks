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

// #[cfg(test)]
// mod tests {
//     use lambdaworks_math::field::{
//         element::FieldElement,
//         fields::{
//             fft_friendly::stark_252_prime_field::Stark252PrimeField, u64_prime_field::U64PrimeField,
//         },
//     };
//     use sha3::{Keccak256, Sha3_256};

//     use crate::merkle_tree::{
//         backends::hash_256_bits::Tree256Bits, merkle::MerkleTree, test_merkle::TestBackend,
//     };

//     type F = Stark252PrimeField;
//     type FE = FieldElement<F>;

//     #[test]
//     // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
//     fn build_merkle_tree_from_an_odd_set_of_leaves() {
//         const MODULUS: u64 = 13;
//         type U64PF = U64PrimeField<MODULUS>;
//         type FE = FieldElement<U64PF>;

//         let values: Vec<FE> = (1..6).map(FE::new).collect();
//         let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values);
//         assert_eq!(merkle_tree.root, FE::new(8));
//     }

//     #[test]
//     fn hash_data_field_element_backend_works_with_keccak() {
//         let values: Vec<FE> = (1..6).map(FE::from).collect();
//         let merkle_tree = MerkleTree::<Tree256Bits<F, Keccak256>>::build(&values);
//         let proof = merkle_tree.get_proof_by_pos(0).unwrap();
//         assert!(proof.verify::<Tree256Bits<F, Keccak256>>(&merkle_tree.root, 0, &values[0]));
//     }

//     #[test]
//     fn hash_data_field_element_backend_works_with_sha3() {
//         let values: Vec<FE> = (1..6).map(FE::from).collect();
//         let merkle_tree = MerkleTree::<Tree256Bits<F, Sha3_256>>::build(&values);
//         let proof = merkle_tree.get_proof_by_pos(0).unwrap();
//         assert!(proof.verify::<Tree256Bits<F, Sha3_256>>(&merkle_tree.root, 0, &values[0]));
//     }
// }
