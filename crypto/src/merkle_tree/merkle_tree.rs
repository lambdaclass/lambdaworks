use std::{cell::RefCell, rc::Rc};
use crate::hash::traits::IsCryptoHash;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

pub struct MerkleTree<F: IsField, H: IsCryptoHash<F>> {
    leafs: Vec<TreeNode<F>>,
    root: TreeNode<F>,
    hasher: H,
}

impl<F: IsField, H: IsCryptoHash<F>> MerkleTree<F, H> {
    pub fn build(values: &[FieldElement<F>], hasher: H) -> MerkleTree<F, H> {
        let mut leafs: Vec<TreeNode<F>> = hash_leafs(values, &hasher);
        let mut level: Vec<TreeNode<F>> = hash_level(&mut leafs, &hasher);

        while level.len() > 1 {
            level = hash_level(&mut level, &hasher);
        }

        return MerkleTree {
            leafs,
            root: level[0].clone(),
            hasher,
        };
    }

    pub fn get_root_hash(self) -> FieldElement<F> {
        self.root.borrow().hash.clone()
    }

    pub fn get_proof(self, value: FieldElement<F>) -> Option<Proof<F>> {
        let hashed_value = self.hasher.hash_one(value.clone());

        if let Some(leaf) = self
            .leafs
            .iter()
            .find(|node| node.borrow().hash == hashed_value)
        {
            let mut merkle_path: Vec<TreeNode<F>> = Vec::new();
            merkle_path = build_merkle_path(leaf.clone(), &mut merkle_path).to_vec();

            return Some(Proof { value, merkle_path });
        }

        return None;
    }
}

fn hash_leafs<F: IsField, H: IsCryptoHash<F>>(
    values: &[FieldElement<F>],
    hasher: &H,
) -> Vec<TreeNode<F>> {
    values
        .iter()
        .map(|value| build_tree_node(hasher.hash_one(value.clone())))
        .collect()
}

fn hash_level<F: IsField, H: IsCryptoHash<F>>(
    values: &mut Vec<TreeNode<F>>,
    hasher: &H,
) -> Vec<TreeNode<F>> {
    if values.len() % 2 != 0 {
        values.push(values[values.len() - 1].clone());
    }

    values
        .chunks(2)
        .map(|chunk| build_parent(chunk[0].clone(), chunk[1].clone(), hasher))
        .collect()
}

fn build_parent<F: IsField, H: IsCryptoHash<F>>(
    left: TreeNode<F>,
    right: TreeNode<F>,
    hasher: &H,
) -> TreeNode<F> {
    let parent_hash = hasher.hash_two(left.borrow().hash.clone(), right.borrow().hash.clone());
    let parent = build_tree_node(parent_hash);

    left.borrow_mut().sibiling = Some(right.clone());
    left.borrow_mut().parent = Some(parent.clone());

    right.borrow_mut().sibiling = Some(left.clone());
    right.borrow_mut().parent = Some(parent.clone());

    return parent;
}

fn build_merkle_path<F: IsField>(
    node: TreeNode<F>,
    merkle_path: &mut Vec<TreeNode<F>>,
) -> &Vec<TreeNode<F>> {
    merkle_path.push(node.clone());

    if let Some(parent) = node.borrow().parent.clone() {
        return build_merkle_path(parent, merkle_path);
    }

    merkle_path
}

pub struct Proof<F: IsField> {
    value: FieldElement<F>,
    merkle_path: Vec<TreeNode<F>>,
}

type TreeNode<F> = Rc<RefCell<Node<F>>>;

fn build_tree_node<F: IsField>(hash: FieldElement<F>) -> TreeNode<F> {
    Rc::new(RefCell::new(Node {
        hash,
        parent: None,
        sibiling: None,
    }))
}

#[derive(Clone, Debug, PartialEq)]
struct Node<F: IsField> {
    hash: FieldElement<F>,
    parent: Option<TreeNode<F>>,
    sibiling: Option<TreeNode<F>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    struct TestHasher;

    impl IsCryptoHash<U64PF> for TestHasher {
        fn hash_one(&self, input: FE) -> FE {
            input + input
        }

        fn hash_two(&self, left: FE, right: FE) -> FE {
            left + right
        }
    }

    #[test]
    fn create_merkle_tree_leafs_from_a_set_of_field_elemnts() {
        let hashed_leafs = hash_leafs(
            &[FE::new(1), FE::new(2), FE::new(3), FE::new(4)].to_vec(),
            &TestHasher,
        );
        let list_of_nodes = [
            build_tree_node(FE::new(2)),
            build_tree_node(FE::new(4)),
            build_tree_node(FE::new(6)),
            build_tree_node(FE::new(8)),
        ]
        .to_vec();

        for (leaf, expected_leaf) in hashed_leafs.iter().zip(list_of_nodes) {
            assert_eq!(leaf.borrow().hash, expected_leaf.borrow_mut().hash);
            assert_eq!(leaf.borrow().parent, None);
            assert_eq!(leaf.borrow().sibiling, None);
        }
    }

    #[test]
    fn apply_hash_level_to_a_set_of_leafs() {
        let hasher = TestHasher;
        let mut hashed_leafs = hash_leafs(
            &[FE::new(1), FE::new(2), FE::new(3), FE::new(4)].to_vec(),
            &hasher,
        );
        let level_one_nodes = hash_level(&mut hashed_leafs, &hasher);
        let expected_list_of_nodes =
            [build_tree_node(FE::new(6)), build_tree_node(FE::new(14))].to_vec();

        for (node, expected_node) in level_one_nodes.iter().zip(expected_list_of_nodes) {
            assert_eq!(node.borrow().hash, expected_node.borrow().hash);
        }

        for (pos, leaf) in hashed_leafs.iter().enumerate() {
            let sibiling_pos;
            let parent_pos;

            if pos % 2 == 0 {
                sibiling_pos = pos + 1;
                parent_pos = pos / 2;
            } else {
                sibiling_pos = pos - 1;
                parent_pos = (pos - 1) / 2;
            }

            assert_eq!(
                leaf.borrow().sibiling.as_ref().unwrap().borrow().hash,
                hashed_leafs[sibiling_pos].borrow().hash
            );
            assert_eq!(
                leaf.borrow().parent.as_ref().unwrap().borrow().hash,
                level_one_nodes[parent_pos].borrow().hash
            );
        }
    }

    #[test]
    fn build_merkle_tree_from_an_even_set_of_leafs() {
        let merkle_tree = MerkleTree::build(
            &[FE::new(1), FE::new(2), FE::new(3), FE::new(4)].to_vec(),
            TestHasher,
        );
        assert_eq!(merkle_tree.root.borrow().hash, FE::new(20));
    }

    #[test]
    fn build_merkle_tree_from_an_odd_set_of_leafs() {
        let merkle_tree = MerkleTree::build(
            &[FE::new(1), FE::new(2), FE::new(3), FE::new(4), FE::new(5)].to_vec(),
            TestHasher,
        );
        assert_eq!(merkle_tree.root.borrow().hash, FE::new(60));
    }

    #[test]
    fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree() {
        let merkle_tree = MerkleTree::build(
            &[FE::new(1), FE::new(2), FE::new(3), FE::new(4), FE::new(5)].to_vec(),
            TestHasher,
        );
        let proof = &merkle_tree.get_proof(FE::new(2)).unwrap();
        let expected_hash = &[
            build_tree_node(FE::new(4)),
            build_tree_node(FE::new(6)),
            build_tree_node(FE::new(7)),
            build_tree_node(FE::new(8)),
        ];
        for (key, elem) in expected_hash.iter().enumerate() {
            assert_eq!(proof.merkle_path[key].borrow().hash, elem.borrow().hash);
        }
    }
}
