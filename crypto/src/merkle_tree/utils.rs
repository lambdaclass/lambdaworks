use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};

use crate::hash::traits::IsHasher;

// pub fn hash_leaves<H: IsHasher>(values: &[H::Type], hasher: &H) -> Vec<H::Type> {
//     values
//         .iter()
//         .map(|val| hasher.hash_leaf(&val))
//         .collect()
// }

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
pub fn complete_until_power_of_two<O: Clone>(values: &mut Vec<O>) -> Vec<O> {
    while !is_power_of_two(values.len()) {
        values.push(values[values.len() - 1].clone())
    }
    values.to_vec()
}

pub fn is_power_of_two(x: usize) -> bool {
    (x != 0) && ((x & (x - 1)) == 0)
}

pub fn build<H: IsHasher>(
    nodes: &mut Vec<H::Type>,
    parent_index: usize,
    hasher: &H,
) -> Vec<H::Type> where H::Type: Clone {
    if is_leaf(nodes.len(), parent_index) {
        return nodes.to_vec();
    }

    let left_child_index = left_child_index(parent_index);
    let right_child_index = right_child_index(parent_index);

    let mut nodes = build(nodes, left_child_index, hasher);
    nodes = build(&mut nodes, right_child_index, hasher);

    nodes[parent_index] = hasher.hash_two(&nodes[left_child_index], &nodes[right_child_index]);
    nodes
}

pub fn is_leaf(lenght: usize, node_index: usize) -> bool {
    (node_index >= (lenght / 2)) && node_index < lenght
}

pub fn left_child_index(parent_index: usize) -> usize {
    parent_index * 2 + 1
}

pub fn right_child_index(parent_index: usize) -> usize {
    parent_index * 2 + 2
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    // use crate::merkle_tree::test_merkle::TestHasher;

    use super::{build, complete_until_power_of_two};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;
    /*
    #[test]
    // expected |2|4|6|8|
    fn hash_leaves_from_a_list_of_field_elemnts() {
        let values: Vec<FE> = (1..5).map(FE::new).collect();
        let hashed_leaves = hash_leaves(&values, &TestHasher);
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
    fn compleate_a_merkle_tree_from_a_set_of_leaves() {
        let leaves: Vec<FE> = (1..9).map(FE::new).collect();

        let mut nodes = vec![FE::zero(); leaves.len() - 1];
        nodes.extend(leaves);

        let tree = build(&mut nodes, ROOT, &TestHasher);
        assert_eq!(tree[ROOT], FE::new(10));
    }
    */
}
