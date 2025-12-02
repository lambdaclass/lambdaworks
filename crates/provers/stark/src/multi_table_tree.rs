use core::hash;
use std::marker::PhantomData;

use digest::{Digest, Output};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

use crate::table::Table;

pub struct MultiTableTree<F, D: Digest, const NUM_BYTES: usize>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    pub root: [u8; NUM_BYTES],
    nodes: Vec<[u8; NUM_BYTES]>,
    _phantom: PhantomData<(F, D)>,
}

impl<F, D: Digest, const NUM_BYTES: usize> MultiTableTree<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    /// This function takes a single row data and converts it to a node.
    fn hash_data(row_data: &[FieldElement<F>]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        for element in row_data.iter() {
            hasher.update(element.as_bytes());
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    /// This function takes a list of data (a list of rows) from which the Merkle
    /// tree will be built from and converts it to a list of leaf nodes.
    fn hash_leaves(unhashed_leaves: &[Vec<FieldElement<F>>]) -> Vec<[u8; NUM_BYTES]> {
        let iter = unhashed_leaves.iter();
        iter.map(|leaf| Self::hash_data(leaf)).collect()
    }

    /// This function takes to children nodes and builds a new parent node.
    /// It will be used in the construction of the Merkle tree.
    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    /// This function takes to children nodes (left and right) and additional data (an other matrix row)
    /// to be injected and builds a new parent node.
    /// It will be used in the construction of the Merkle tree.
    ///
    /// TODO. Ask which option we should do:
    /// 1. H(L, R, new)
    /// 2. H(L, R, H(new))
    /// 3. H(H(L, R), H(new))
    fn hash_new_parent_with_injection(
        left: &[u8; NUM_BYTES],
        right: &[u8; NUM_BYTES],
        data_to_inject: &[FieldElement<F>],
    ) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();

        hasher.update(left);
        hasher.update(right);

        // // Option 1.
        // hasher.update(data_to_inject.as_bytes());

        // Option 2.
        let hashed_injection = Self::hash_data(data_to_inject);
        hasher.update(hashed_injection);

        // Option 3.
        // ...

        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    use sha3::Sha3_256;

    use super::MultiTableTree;

    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn run_hash_data() {
        let leaves_data = [
            vec![FE::from(1u64), FE::from(10u64)],
            vec![FE::from(2u64), FE::from(20u64)],
            vec![FE::from(3u64), FE::from(30u64)],
            vec![FE::from(4u64), FE::from(40u64)],
            vec![FE::from(5u64), FE::from(50u64)],
            vec![FE::from(6u64), FE::from(60u64)],
            vec![FE::from(7u64), FE::from(70u64)],
            vec![FE::from(8u64), FE::from(80u64)],
        ];

        let leaves_hashed = MultiTableTree::<F, Sha3_256, 32>::hash_leaves(&leaves_data);
        println!("Leaves hashed: {:?}", leaves_hashed);
    }

    #[test]
    fn run_hash_new_parent_with_injection() {
        let leaves_data = [
            vec![FE::from(1u64), FE::from(10u64)],
            vec![FE::from(2u64), FE::from(20u64)],
            vec![FE::from(3u64), FE::from(30u64)],
            vec![FE::from(4u64), FE::from(40u64)],
            vec![FE::from(5u64), FE::from(50u64)],
            vec![FE::from(6u64), FE::from(60u64)],
            vec![FE::from(7u64), FE::from(70u64)],
            vec![FE::from(8u64), FE::from(80u64)],
        ];

        let leaves_hashed = MultiTableTree::<F, Sha3_256, 32>::hash_leaves(&leaves_data);

        let data_to_inject = &[FE::from(9u64), FE::from(90u64)];

        let parent_hash = MultiTableTree::<F, Sha3_256, 32>::hash_new_parent_with_injection(
            &leaves_hashed[0],
            &leaves_hashed[1],
            data_to_inject,
        );

        println!("Parent hash with injection: {:?}", parent_hash);
    }
}
