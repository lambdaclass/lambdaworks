use crate::hash::traits::IsCryptoHash;
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::{element::FieldElement, traits::IsField},
};
use std::{cell::RefCell, rc::Rc};

pub struct MerkleTree<F: IsField, H: IsCryptoHash<F>> {
    pub leafs: Vec<TreeNode<F>>,
    pub root: TreeNode<F>,
    hasher: H,
}

#[derive(Debug, Clone)]
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

impl<F: IsField, H: IsCryptoHash<F> + Clone> MerkleTree<F, H> {
    pub fn build(values: &[FieldElement<F>]) -> MerkleTree<F, H> {
        let hasher = H::new();
        let mut leafs: Vec<TreeNode<F>> = hash_leafs(values, &hasher);
        let mut level: Vec<TreeNode<F>> = hash_level(&mut leafs, &hasher);

        while level.len() > 1 {
            level = hash_level(&mut level, &hasher);
        }

        MerkleTree {
            leafs,
            root: level[0].clone(),
            hasher,
        }
    }

    pub fn get_root_hash(&self) -> FieldElement<F> {
        self.root.borrow().hash.clone()
    }

    pub fn get_proof(&self, value: FieldElement<F>) -> Option<Proof<F, H>> {
        let hashed_value = self.hasher.hash_one(value.clone());

        if let Some(leaf) = self
            .leafs
            .iter()
            .find(|node| node.borrow().hash == hashed_value)
        {
            let merkle_path = build_merkle_path(leaf.clone(), &mut Vec::new());

            return Some(Proof {
                value,
                merkle_path,
                hasher: self.hasher.clone(),
            });
        }

        None
    }

    pub fn get_proof_by_pos(&self, pos: usize, value: &FieldElement<F>) -> Option<Proof<F, H>> {
        let hash_leaf = self.hasher.hash_one(value.clone());

        if let Some(leaf) = self.leafs.get(pos) {
            if hash_leaf != leaf.borrow().hash {
                return None;
            }

            let merkle_path = build_merkle_path(leaf.clone(), &mut Vec::new());

            return Some(Proof {
                value: value.clone(),
                merkle_path,
                hasher: self.hasher.clone(),
            });
        }

        None
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

    right.borrow_mut().sibiling = Some(left);
    right.borrow_mut().parent = Some(parent.clone());

    parent
}

fn build_merkle_path<F: IsField>(
    node: TreeNode<F>,
    merkle_path: &mut Vec<TreeNode<F>>,
) -> Vec<TreeNode<F>> {
    merkle_path.push(node.clone());

    if let Some(parent) = node.borrow().parent.clone() {
        return build_merkle_path(parent, merkle_path).to_vec();
    }

    merkle_path.to_vec()
}

#[derive(Debug, Clone)]
pub struct Proof<F: IsField, H: IsCryptoHash<F>> {
    pub value: FieldElement<F>,
    merkle_path: Vec<TreeNode<F>>,
    hasher: H,
}

impl<F: IsField, H: IsCryptoHash<F>> Proof<F, H> {
    pub fn verify(&self, root_hash: FieldElement<F>) -> bool {
        let mut hashed_value = self.hasher.hash_one(self.value.clone());

        for node in self.merkle_path.iter() {
            if let Some(sibiling) = &node.borrow().sibiling {
                hashed_value = self
                    .hasher
                    .hash_two(hashed_value, sibiling.borrow().hash.clone());
            }
        }
        hashed_value == root_hash
    }
}

pub type TreeNode<F> = Rc<RefCell<Node<F>>>;

fn build_tree_node<F: IsField>(hash: FieldElement<F>) -> TreeNode<F> {
    Rc::new(RefCell::new(Node {
        hash,
        parent: None,
        sibiling: None,
    }))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node<F: IsField> {
    pub hash: FieldElement<F>,
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
    fn create_merkle_tree_leafs_from_a_set_of_field_elemnts() {
        let hashed_leafs = hash_leafs(
            &[FE::new(1), FE::new(2), FE::new(3), FE::new(4)],
            &TestHasher,
        );
        let list_of_nodes = [
            build_tree_node(FE::new(2)),
            build_tree_node(FE::new(4)),
            build_tree_node(FE::new(6)),
            build_tree_node(FE::new(8)),
        ];

        for (leaf, expected_leaf) in hashed_leafs.iter().zip(list_of_nodes) {
            assert_eq!(leaf.borrow().hash, expected_leaf.borrow_mut().hash);
            assert_eq!(leaf.borrow().parent, None);
            assert_eq!(leaf.borrow().sibiling, None);
        }
    }

    #[test]
    fn apply_hash_level_to_a_set_of_leafs() {
        let hasher = TestHasher;
        let mut hashed_leafs =
            hash_leafs(&[FE::new(1), FE::new(2), FE::new(3), FE::new(4)], &hasher);
        let level_one_nodes = hash_level(&mut hashed_leafs, &hasher);
        let expected_list_of_nodes = [build_tree_node(FE::new(6)), build_tree_node(FE::new(14))];

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
        let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&[
            FE::new(1),
            FE::new(2),
            FE::new(3),
            FE::new(4),
        ]);
        assert_eq!(merkle_tree.root.borrow().hash, FE::new(20));
    }

    #[test]
    fn build_merkle_tree_from_an_odd_set_of_leafs() {
        let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&[
            FE::new(1),
            FE::new(2),
            FE::new(3),
            FE::new(4),
            FE::new(5),
        ]);
        assert_eq!(merkle_tree.root.borrow().hash, FE::new(60));
    }

    #[test]
    fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree() {
        let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&[
            FE::new(1),
            FE::new(2),
            FE::new(3),
            FE::new(4),
            FE::new(5),
        ]);
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
        let proof = &merkle_tree.get_proof_by_pos(1, &FE::new(2)).unwrap();
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

    // #[test]
    // fn verify_a_proof_over_value_that_belongs_to_a_given_merkle_tree() {
    //     let merkle_tree = MerkleTree::<U64PF, TestHasher>::build(&[
    //         FE::new(1),
    //         FE::new(2),
    //         FE::new(3),
    //         FE::new(4),
    //         FE::new(5),
    //     ]);
    //     let proof = merkle_tree.get_proof(FE::new(2)).unwrap();

    //     assert!(MerkleTree::verify(&proof, merkle_tree.get_root_hash()));
    // }
}
