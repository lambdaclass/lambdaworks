// Inlfuenced https://github.com/melekes/merkle-tree-rs
use crate::hash::traits::IsCryptoHash;
use lambdaworks_math::field::{traits::IsField, element::FieldElement};
//1. I recive a vecotr of fields of length L

//2. Applied a 1to1 Hash on Each Field  
//   If L isn't a multiple of 2 repeat the last elemnt until is a multiple of 2.

//3. If L is a multiple of 2 I hash each pair of consecutive hashed field

//4. Repeat (3) until you have only one element

// pub struct MerkleTree<T, H> {
//     pub nodes: Vec<T>,
//     hasher: H,
// } where T: field::IsField, H: IsCryptoHash;

// impl MerkleTree{

//     pub fn build<T: field::IsField, H: IsCryptoHash>(values: &[T], mut hasher: H) -> Result<MerkleTree<T, H>>  {
//         let hashed_leafs: Vec<Hash> = hash_leaf_nodes(values, hasher);
//         let internal_nodes: Vec<Hash> = hash_internal_nodes(hashed_leafs, hasher);
//         return MerkleTree{nodes: internal_node.co}


//     }

// }


fn hash_leafs<F: IsField, H: IsCryptoHash<F>>(mut values: Vec<FieldElement<F>>) -> Vec<FieldElement<F>> {

    // If the list of valuese is not a multiple of 2, add a copy of the last element to complete the tree
    if values.len() % 2 != 0 {
        values.push(values[values.len()-1].clone())
    }

    values.iter().map(|val| H::hash_one(val.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use lambdaworks_math::field::{fields::u64_prime_field::U64PrimeField, element::FieldElement};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    struct TestHasher;

    impl IsCryptoHash<U64PF> for TestHasher{
        fn hash_one(input: FE) -> FE {
            input + input
        }

        fn hash_two(
            left: FE,
            right: FE,
        ) -> FE {
            left + right
        }
    }

    #[test]
    fn hash_leafs_of_an_even_set_of_values() {
        let hashed_leafs = super::hash_leafs::<U64PF, TestHasher>([FE::new(1), FE::new(2), FE::new(3), FE::new(4)].to_vec());
        assert_eq!(hashed_leafs, [FE::new(2), FE::new(4), FE::new(6), FE::new(8)].to_vec());
    }

    #[test]
    fn hash_leafs_of_an_edd_set_of_values() {
        let hashed_leafs = super::hash_leafs::<U64PF, TestHasher>([FE::new(1), FE::new(2), FE::new(3)].to_vec());
        assert_eq!(hashed_leafs, [FE::new(2), FE::new(4), FE::new(6), FE::new(6)].to_vec());
    }
}
