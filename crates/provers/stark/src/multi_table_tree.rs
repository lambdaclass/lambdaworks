use core::cmp::Reverse;
use itertools::Itertools;
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
    /// Create a Merkle tree from a slice of tables.
    /// Each table must have a power of two number of rows.
    pub fn build(tables: &[Table<F>]) -> Option<Self> {
        let mut sorted_tables = tables
            .iter()
            .sorted_by_key(|t| Reverse(t.height))
            .peekable();

        let Some(first) = sorted_tables.peek() else {
            return None;
        };
        let max_height = first.height;
        // Podemos calcular la cantidad de nodos totales como 2 * Leafs - 1.
        let mut nodes = Vec::with_capacity(2 * max_height - 1);

        let max_height_tables: Vec<_> = sorted_tables
            .peeking_take_while(|t| t.height == max_height)
            .collect();

        for row_idx in 0..max_height {
            let concatenated_row: Vec<&FieldElement<F>> = max_height_tables
                .iter()
                .flat_map(|table| table.get_row(row_idx))
                .collect();

            let mut hasher = D::new();
            for element in concatenated_row {
                hasher.update(element.as_bytes());
            }
            let hash: [u8; NUM_BYTES] = hasher.finalize().into();

            nodes.push(hash);
        }

        let mut current_layer_size = max_height;
        let mut current_layer_start = 0;

        while current_layer_size > 1 {
            let next_layer_size = current_layer_size / 2;

            let next_layer_tables: Option<Vec<_>> = sorted_tables.peek().and_then(|next| {
                if next.height == next_layer_size {
                    Some(
                        sorted_tables
                            .peeking_take_while(|t| t.height == next_layer_size)
                            .collect(),
                    )
                } else {
                    None
                }
            });

            for i in 0..next_layer_size {
                let left_child = &nodes[current_layer_start + 2 * i];
                let right_child = &nodes[current_layer_start + 2 * i + 1];

                let hash = if let Some(ref tables) = next_layer_tables {
                    hash_with_table_row(left_child, right_child, tables, i)
                } else {
                    hash_two_nodes(left_child, right_child)
                };

                nodes.push(hash);
            }

            current_layer_start += current_layer_size;
            current_layer_size = next_layer_size;
        }

        println!("Nodes: {:?}", nodes);
        Some(MultiTableTree {
            root: nodes.last().unwrap().clone(),
            nodes,
            _phantom: PhantomData::<(F, D)>,
        })
    }

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
    use super::*;
    use crate::table::Table;
    use lambdaworks_math::field::fields::fft_friendly::babybear_u32::Babybear31PrimeField;
    use sha3::Keccak256;

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn build_tree() {
        let data1: Vec<_> = (0..24).map(|i| FE::from(i)).collect();
        let table1 = Table::new(data1, 3);

        let data2: Vec<_> = (100..116).map(|i| FE::from(i)).collect();
        let table2 = Table::new(data2, 2);

        let data3: Vec<_> = (200..216).map(|i| FE::from(i)).collect();
        let table3 = Table::new(data3, 4);

        let merkle_tree =
            MultiTableTree::<F, Keccak256, 32>::build(&[table1, table2, table3]).unwrap();
    }

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
