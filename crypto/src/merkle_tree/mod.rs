use crate::hash::traits::IsCryptoHash;
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::{element::FieldElement, traits::IsField},
};

pub struct MerkleTree<F: IsField, H: IsCryptoHash<F>> {
    pub root: FieldElement<F>,
    nodes: Vec<FieldElement<F>>,
    hasher: H,
}

const ROOT: usize = 0;

impl<F: IsField, H: IsCryptoHash<F> + Clone> MerkleTree<F, H> {
    pub fn build(values: &[FieldElement<F>]) -> MerkleTree<F, H> {
        let hasher = H::new();
        let mut nodes: Vec<FieldElement<F>> = hash_leafs(values, &hasher);

        //The leaf must be a power of 2 set
        nodes = complete_until_power_of_two(&mut nodes);

        //There lenght of leafs minus one inner nodes in the merkle tree
        let mut inner_nodes = vec![FieldElement::zero(); nodes.len() - 1];
        inner_nodes.extend(nodes);

        //Build the inner nodes of the tree
        let nodes = build(&mut inner_nodes, ROOT, &hasher);

        MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
            hasher,
        }
    }

    fn get_leafs(&self) -> Vec<FieldElement<F>> {
        let leaves_start = self.nodes.len() / 2;
        Vec::from(&self.nodes[leaves_start..])
    }

    pub fn get_proof(&self, value: &FieldElement<F>) -> Option<Proof<F, H>> {
        let hashed_leaf = self.hasher.hash_one(value.clone());

        if let Some(mut pos) = self
            .get_leafs()
            .iter()
            .position(|leaf| *leaf == hashed_leaf)
        {
            pos += self.nodes.len() / 2;

            let merkle_path = self.build_merkle_path(pos);

            return self.create_proof(merkle_path, value);
        }

        None
    }

    pub fn get_proof_by_pos(&self, pos: usize, value: FieldElement<F>) -> Option<Proof<F, H>> {
        let hashed_leaf = self.hasher.hash_one(value.clone());

        let pos = pos + self.nodes.len() / 2;

        if self.nodes[pos] != hashed_leaf {
            return None;
        }

        let merkle_path = self.build_merkle_path(pos);

        self.create_proof(merkle_path, &value)
    }

    fn create_proof(
        &self,
        merkle_path: Vec<(FieldElement<F>, bool)>,
        value: &FieldElement<F>,
    ) -> Option<Proof<F, H>> {
        Some(Proof {
            value: value.clone(),
            merkle_path,
            hasher: self.hasher.clone(),
        })
    }

    fn build_merkle_path(&self, pos: usize) -> Vec<(FieldElement<F>, bool)> {
        let mut merkle_path = Vec::new();
        let mut pos = pos;

        while pos != ROOT {
            merkle_path.push((self.nodes[sibling_index(pos)].clone(), pos % 2 == 0));
            pos = parent_index(pos);
        }

        merkle_path
    }
    pub fn verify(proof: &Proof<F, H>, root_hash: FieldElement<F>) -> bool {
        let mut hashed_value = proof.hasher.hash_one(proof.value.clone());

        for (sibiling_node, is_left) in proof.merkle_path.iter().rev() {
            if *is_left {
                hashed_value = proof.hasher.hash_two(hashed_value, sibiling_node.clone());
            } else {
                hashed_value = proof.hasher.hash_two(sibiling_node.clone(), hashed_value);
            }
        }

        root_hash == hashed_value
    }
}

fn sibling_index(node_index: usize) -> usize {
    if node_index % 2 == 0 {
        node_index - 1
    } else {
        node_index + 1
    }
}

fn parent_index(node_index: usize) -> usize {
    if node_index % 2 == 0 {
        (node_index - 1) / 2
    } else {
        node_index / 2
    }
}
fn hash_leafs<F: IsField, H: IsCryptoHash<F>>(
    values: &[FieldElement<F>],
    hasher: &H,
) -> Vec<FieldElement<F>> {
    values
        .iter()
        .map(|val| hasher.hash_one(val.clone()))
        .collect()
}

// The list of values is completed repeating the last value to a power of two length
fn complete_until_power_of_two<F: IsField>(
    values: &mut Vec<FieldElement<F>>,
) -> Vec<FieldElement<F>> {
    while !is_power_of_two(values.len()) {
        values.push(values[values.len() - 1].clone())
    }
    values.to_vec()
}

fn is_power_of_two(x: usize) -> bool {
    (x != 0) && ((x & (x - 1)) == 0)
}

fn build<F: IsField, H: IsCryptoHash<F>>(
    nodes: &mut Vec<FieldElement<F>>,
    parent_index: usize,
    hasher: &H,
) -> Vec<FieldElement<F>> {
    if is_leaf(nodes.len(), parent_index) {
        return nodes.to_vec();
    }

    let left_child_index = left_child_index(parent_index);
    let right_child_index = right_child_index(parent_index);

    let mut nodes = build(nodes, left_child_index, hasher);
    nodes = build(&mut nodes, right_child_index, hasher);

    nodes[parent_index] = hasher.hash_two(
        nodes[left_child_index].clone(),
        nodes[right_child_index].clone(),
    );
    nodes
}

fn is_leaf(lenght: usize, node_index: usize) -> bool {
    (node_index >= (lenght / 2)) && node_index < lenght
}

fn left_child_index(parent_index: usize) -> usize {
    parent_index * 2 + 1
}

fn right_child_index(parent_index: usize) -> usize {
    parent_index * 2 + 2
}

pub struct Proof<F: IsField, H: IsCryptoHash<F>> {
    value: FieldElement<F>,
    merkle_path: Vec<(FieldElement<F>, bool)>,
    hasher: H,
}

#[derive(Debug)]
pub struct DefaultHasher;

impl<F: IsField> IsCryptoHash<F> for DefaultHasher {
    fn new() -> DefaultHasher {
        DefaultHasher
    }

    fn hash_one(&self, input: FieldElement<F>) -> FieldElement<F> {
        &input + &input
    }

    fn hash_two(&self, left: FieldElement<F>, right: FieldElement<F>) -> FieldElement<F> {
        left + right
    }
}

pub type MerkleTreeDefault = MerkleTree<BLS12381PrimeField, DefaultHasher>;

#[cfg(test)]
mod tests {
    use super::*;

    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    #[derive(Clone)]
    struct TestHasher;

    impl IsCryptoHash<U64PF> for TestHasher {
        fn hash_one(&self, input: FE) -> FE {
            input + input
        }

        fn hash_two(&self, left: FE, right: FE) -> FE {
            left + right
        }

        fn new() -> TestHasher {
            TestHasher
        }
    }

    #[test]
    fn hash_leafs_from_a_list_of_field_elemnts() {
        let hashed_leafs = hash_leafs(
            &[FE::new(1), FE::new(2), FE::new(3), FE::new(4)],
            &TestHasher,
        );

        let list_of_nodes = &[FE::new(2), FE::new(4), FE::new(6), FE::new(8)];

        for (leaf, expected_leaf) in hashed_leafs.iter().zip(list_of_nodes) {
            assert_eq!(leaf, expected_leaf);
        }
    }

    #[test]
    fn complete_the_length_of_a_list_of_fields_elemnts_to_be_a_power_of_two() {
        let mut values = [FE::new(2), FE::new(4), FE::new(6), FE::new(8), FE::new(10)].to_vec();
        let hashed_leafs = complete_until_power_of_two(&mut values);

        let list_of_nodes = &[
            FE::new(2),
            FE::new(4),
            FE::new(6),
            FE::new(8),
            FE::new(10),
            FE::new(10),
            FE::new(10),
            FE::new(10),
        ];

        for (leaf, expected_leaf) in hashed_leafs.iter().zip(list_of_nodes) {
            assert_eq!(leaf, expected_leaf);
        }
    }

    //| 10 | 3 | 7 | 1 | 2 | 3 | 4 |
    #[test]
    fn build_merkle_tree_from_a_power_of_two_list_of_values() {
        let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&[
            FE::new(1),
            FE::new(2),
            FE::new(3),
            FE::new(4),
        ]);
        assert_eq!(merkle_tree.root, FE::new(20));
    }

    // | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    #[test]
    fn build_merkle_tree_from_an_odd_set_of_leafs() {
        let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&[
            FE::new(1),
            FE::new(2),
            FE::new(3),
            FE::new(4),
            FE::new(5),
        ]);
        assert_eq!(merkle_tree.root, FE::new(8));
    }

    // | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    #[test]
    fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree_when_given_the_leaf_position()
    {
        let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&[
            FE::new(1),
            FE::new(2),
            FE::new(3),
            FE::new(4),
            FE::new(5),
        ]);
        let proof = &merkle_tree.get_proof_by_pos(1, FE::new(2)).unwrap();
        for ((node, _), expected_node) in
            proof
                .merkle_path
                .iter()
                .zip(&[FE::new(2), FE::new(1), FE::new(1)])
        {
            assert_eq!(node, expected_node);
        }

        assert!(MerkleTree::verify(&proof, merkle_tree.root));
    }

    #[test]
    fn verify_a_proof_over_value_that_belongs_to_a_given_merkle_tree() {
        let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&[
            FE::new(1),
            FE::new(2),
            FE::new(3),
            FE::new(4),
            FE::new(5),
        ]);

        let proof = merkle_tree.get_proof(&FE::new(2)).unwrap();

        for ((node, _), expected_node) in
            proof
                .merkle_path
                .iter()
                .zip(&[FE::new(2), FE::new(1), FE::new(1)])
        {
            assert_eq!(node, expected_node);
        }

        assert!(MerkleTree::verify(&proof, merkle_tree.root));
    }
}
