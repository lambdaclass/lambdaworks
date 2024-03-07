use alloc::vec::Vec;

use super::traits::IsMerkleTreeBackend;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn sibling_index(node_index: usize) -> usize {
    if node_index % 2 == 0 {
        node_index - 1
    } else {
        node_index + 1
    }
}

pub fn parent_index(node_index: usize) -> usize {
    if node_index % 2 == 0 {
        (node_index - 1) / 2
    } else {
        node_index / 2
    }
}

// The list of values is completed repeating the last value to a power of two length
pub fn complete_until_power_of_two<T: Clone>(values: &mut Vec<T>) -> Vec<T> {
    while !is_power_of_two(values.len()) {
        values.push(values[values.len() - 1].clone())
    }
    values.to_vec()
}

pub fn is_power_of_two(x: usize) -> bool {
    (x != 0) && ((x & (x - 1)) == 0)
}

// ! CAUTION !
// Make sure n=nodes.len()+1 is a power of two, and the last n/2 elements (leaves) are populated with hashes.
// This function takes no precautions for other cases.
pub fn build<B: IsMerkleTreeBackend>(nodes: &mut [B::Node], leaves_len: usize)
where
    B::Node: Clone,
{
    let mut level_begin_index = leaves_len - 1;
    let mut level_end_index = 2 * level_begin_index;
    loop {
        let new_level_begin_index = level_begin_index / 2;
        let new_level_length = level_begin_index - new_level_begin_index;

        let (new_level_iter, children_iter) =
            nodes[new_level_begin_index..level_end_index + 1].split_at_mut(new_level_length);

        #[cfg(feature = "parallel")]
        let parent_and_children_zipped_iter = new_level_iter
            .into_par_iter()
            .zip(children_iter.par_chunks_exact(2));
        #[cfg(not(feature = "parallel"))]
        let parent_and_children_zipped_iter =
            new_level_iter.iter_mut().zip(children_iter.chunks_exact(2));

        parent_and_children_zipped_iter.for_each(|(new_parent, children)| {
            *new_parent = B::hash_new_parent(&children[0], &children[1]);
        });

        level_end_index = level_begin_index - 1;
        level_begin_index = new_level_begin_index;

        if level_begin_index == level_end_index {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{test_merkle::TestBackend, traits::IsMerkleTreeBackend};

    use super::{build, complete_until_power_of_two};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;
    #[test]
    // expected |2|4|6|8|
    fn hash_leaves_from_a_list_of_field_elemnts() {
        let values: Vec<FE> = (1..5).map(FE::new).collect();
        let hashed_leaves = TestBackend::hash_leaves(&values);
        let list_of_nodes = &[FE::new(2), FE::new(4), FE::new(6), FE::new(8)];
        for (leaf, expected_leaf) in hashed_leaves.iter().zip(list_of_nodes) {
            assert_eq!(leaf, expected_leaf);
        }
    }

    #[test]
    // expected |1|2|3|4|5|5|5|5|
    fn complete_the_length_of_a_list_of_fields_elements_to_be_a_power_of_two() {
        let mut values: Vec<FE> = (1..6).map(FE::new).collect();
        let hashed_leaves = complete_until_power_of_two(&mut values);

        let mut expected_leaves = (1..6).map(FE::new).collect::<Vec<FE>>();
        expected_leaves.extend([FE::new(5); 3]);

        for (leaf, expected_leaves) in hashed_leaves.iter().zip(expected_leaves) {
            assert_eq!(*leaf, expected_leaves);
        }
    }

    const ROOT: usize = 0;

    #[test]
    // expected |10|10|13|3|7|11|2|1|2|3|4|5|6|7|8|
    fn complete_a_merkle_tree_from_a_set_of_leaves() {
        let leaves: Vec<FE> = (1..9).map(FE::new).collect();
        let leaves_len = leaves.len();

        let mut nodes = vec![FE::zero(); leaves.len() - 1];
        nodes.extend(leaves);

        build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
        assert_eq!(nodes[ROOT], FE::new(10));
    }
}
