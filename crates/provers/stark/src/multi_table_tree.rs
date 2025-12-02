use core::cmp::Reverse;
use itertools::Itertools;
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
}
