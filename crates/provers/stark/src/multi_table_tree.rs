use core::cmp::Reverse;
use itertools::Itertools;
use std::marker::PhantomData;

use digest::{Digest, Output};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

use crate::table::Table;

//                                   H(L|R)
//                            /                     \
//                    H(L|R)                         H(L|R)
//                /         \                      /         \
//      H(L|R|H(o0))     H(L|R|H(o1))   H(L|R|H(o2))   H(L|R|H(o3))
//       /       \                                       /          \
//  H(m0|n0)   H(m2|n2)             ...               H(m6|n6)       H(m7|n7)

// Nodes: [H(m0‖n0), H(m2‖n2), ... , H(l|R)]
// injected_leaves: < [H(o0), H(01), H(o2), H(o3)] >


// TODO. Se tendría que poder usar para hashes que devuelvan field elements.
pub struct MultiTableTree<F, D: Digest, const NUM_BYTES: usize>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    pub root: [u8; NUM_BYTES],
    // TODO: Los nodos están ordenados de izquierda a derecha y de abajo hacia arriba.
    // Ver si lo sincronizamos con MerkleTree.

    // Nodos de abajo hacia arriba. Incluye las leaves (que vienen de las tablas más altas).
    nodes: Vec<[u8; NUM_BYTES]>,
    // Son los hashes de las filas de las tablas más bajas.
    // están separados por lrgo de las tablas.
    // Tener en cuenta que si hay varias tablas de la misma altura, se inyectan todas sus filas concatenadas.
    // O sea que acá se guarda el hash de las filas concatenadas.
    injected_leaves: Vec<Vec<[u8; NUM_BYTES]>>,
    _phantom: PhantomData<(F, D)>,
}

#[derive(Debug)]
pub enum MultiTableTreeError {
    EmptyTree,
    WrongHeight,
}


impl<F, D: Digest, const NUM_BYTES: usize> MultiTableTree<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    /// Create a Merkle tree from a slice of tables.
    /// Each table must have a power of two number of rows.
    pub fn build(tables: &[Table<F>]) -> Result<Self, MultiTableTreeError> {

        // Check all tables have a power of two number of rows.
        if tables.into_iter().any(|t| !t.height.is_power_of_two()) {
            return Err(MultiTableTreeError::WrongHeight);
        };

        // TODO: Ver que pasa con el orden de tablas de igual height.
        let mut sorted_tables = tables
            .iter()
            .sorted_by_key(|t| Reverse(t.height))
            .peekable();

        let Some(first) = sorted_tables.peek() else {
            return Err(MultiTableTreeError::EmptyTree);
        };
        let max_height = first.height;
        
        // Podemos calcular la cantidad de nodos totales como 2 * Leafs - 1.
        let mut nodes = Vec::with_capacity(2 * max_height - 1);

        // TODO: Ver si se puede dejar como iteradores en vez de collect.
        let max_height_tables: Vec<_> = sorted_tables
            .peeking_take_while(|t| t.height == max_height)
            .collect();

        for row_idx in 0..max_height {
            let concatenated_row: Vec<&FieldElement<F>> = max_height_tables
                .iter()
                .flat_map(|table| table.get_row(row_idx))
                .collect();

            let hash = Self::hash_data(concatenated_row);
            nodes.push(hash);
        }

        let mut current_layer_size = max_height;
        let mut current_layer_start = 0;

        let mut injected_leaves = Vec::new();

        while current_layer_size > 1 {

            let mut injected_leaves_for_this_layer = Vec::new();

            let next_layer_size = current_layer_size / 2;

            // TODO: Ver si es necesario tener esto
            let has_tables_to_inject = sorted_tables
                .peek()
                .is_some_and(|next| next.height == next_layer_size);

            let next_layer_tables = if has_tables_to_inject {
                Some(
                    sorted_tables
                        .peeking_take_while(|t| t.height == next_layer_size)
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };

            for i in 0..next_layer_size {
                let left_child = &nodes[current_layer_start + 2 * i];
                let right_child = &nodes[current_layer_start + 2 * i + 1];

                // TODO: No hace falta chequear todos los nodos, se puede hacer algo para que solo chquee una vez con 'has_tables_to_inject'. 
                let hash = if let Some(ref tables) = next_layer_tables {
                    let concatenated_row: Vec<&FieldElement<F>> =
                        tables.iter().flat_map(|table| table.get_row(i)).collect();
                    let hash_to_inject = Self::hash_data(concatenated_row);
                    injected_leaves_for_this_layer.push(hash_to_inject); 
                    Self::hash_new_parent_with_injection(left_child, right_child, &hash_to_inject)
                } else {
                    Self::hash_new_parent(left_child, right_child)
                };

                nodes.push(hash);
            }

            current_layer_start += current_layer_size;
            current_layer_size = next_layer_size;

            if injected_leaves_for_this_layer.len() > 0 {
                injected_leaves.push(injected_leaves_for_this_layer);
            } 
        }

        Ok(MultiTableTree {
            root: nodes.last().unwrap().clone(),
            nodes,
            injected_leaves,
            _phantom: PhantomData::<(F, D)>,
        })
    }

    // /// Generates the proof for a specific leaf index.
    // fn build_proof(&self, index: usize) -> MultiTableProof<F, D, NUM_BYTES> {
    //     // leaves = (len + 1) / 2 
    //     let max_height = (self.nodes.len() + 1) / 2;

    //     let mut merkle_path = Vec::new();
    //     let mut current_index = index;
    //     let mut current_layer_start = 0;
    //     let mut current_layer_size = max_height;

    //     // move up to the root
    //     // refactor to use functions already created
    //     while current_layer_size > 1 {
    //         let sibling_index = if current_index % 2 == 0 {
    //             current_index + 1
    //         } else {
    //             current_index - 1
    //         };

    //         merkle_path.push(self.nodes[current_layer_start + sibling_index]);

    //         current_layer_start += current_layer_size;
    //         current_layer_size /= 2;
    //         current_index /= 2;
    //     }
    //     MultiTableProof { merkle_path, _phantom }
    // }

    /// This function takes a single row data and converts it to a node.
    fn hash_data(row_data: Vec<&FieldElement<F>>) -> [u8; NUM_BYTES] {
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
    // fn hash_leaves(unhashed_leaves: &[Vec<FieldElement<F>>]) -> Vec<[u8; NUM_BYTES]> {
    //     let iter = unhashed_leaves.iter();
    //     iter.map(|leaf| Self::hash_data(leaf)).collect()
    // }

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
    /// 1. H(L, R, new) -> Más barata. Ver seguridad.
    /// 2. H(L, R, H(new)) -> Hasheamos las leaves
    /// 3. H(H(L, R), H(new)) -> Más parecido a plonky 3 (pero con el compress)
    fn hash_new_parent_with_injection(
        left: &[u8; NUM_BYTES],
        right: &[u8; NUM_BYTES],
        hash_to_inject: &[u8; NUM_BYTES],
    ) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();

        hasher.update(left);
        hasher.update(right);
        hasher.update(hash_to_inject);

        // // Option 1.
        // hasher.update(data_to_inject.as_bytes());

        // Option 2.
        // let hashed_injection = Self::hash_data(data_to_inject);
        // hasher.update(hashed_injection);

        // Option 3.
        // ...

        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiTableProof<F, D: Digest, const NUM_BYTES: usize>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    pub merkle_path: Vec<Vec<[u8; NUM_BYTES]>>,
    _phantom: PhantomData<(F, D)>,
}
    
pub struct ValueByLayer<F: IsField> {
    value: Vec<F>,
    layer: usize,
}

impl <F, D, const NUM_BYTES: usize> MultiTableProof<F, D, NUM_BYTES> 
where
    F: IsField,
    D: Digest,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,

{

    // /// Verifies a Merkle inclusion proof for the value contained at leaf index.
    // pub fn verify(&self, root_hash: [u8; NUM_BYTES], mut index: usize, values: Vec<ValueByLayer<F>>) -> bool
    // {
    //     let number_of_layers = self.merkle_path.len();

    //     let mut hashed_value = B::hash_data(values);

    //     for sibling_node in self.merkle_path.iter() {
    //         if index.is_multiple_of(2) {
    //             hashed_value = B::hash_new_parent(&hashed_value, sibling_node);
    //         } else {
    //             hashed_value = B::hash_new_parent(sibling_node, &hashed_value);
    //         }

    //         index >>= 1;
    //     }

    //     root_hash == &hashed_value
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Table;
    use lambdaworks_math::field::fields::fft_friendly::{babybear_u32::Babybear31PrimeField};
    use proptest::result;
    use sha3::Keccak256;

    use rand::random;

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    // Helper function to create a base field table for testing
    fn create_random_table(height: usize, width: usize) -> Table<F> {
        let data_size = width * height;
        let data = (0..data_size).map(|_| FieldElement::<F>::from(random::<u64>())).collect();
        Table { data, width, height }
    }

    #[test]
    fn test_build_empty_list_of_table() {
        let tables: Vec<Table<F>> = vec![];
        let tree = MultiTableTree::<F, Keccak256, 32>::build(&tables);
        assert!(matches!(
            tree,
            Err(MultiTableTreeError::EmptyTree),
        ));
    }

    #[test]
    fn test_build_one_empty_table() {
        let table_1 = create_random_table(2, 1);
        let table_2 = create_random_table(16, 0); // Empty table with 0 columns.
        let table_3 = create_random_table(8, 2);
        let tree = MultiTableTree::<F, Keccak256, 32>::build(&[table_1, table_2, table_3]);
        assert!(matches!(
            tree,
            Err(MultiTableTreeError::WrongHeight),
        ));
    }

    #[test]
    fn test_build_single_table_one_row() {
        // Table size: 1 row, 5 columns.
        let table = create_random_table(1, 5);
        let tree = MultiTableTree::<F, Keccak256, 32>::build(&[table]).unwrap();
        assert_eq!(tree.nodes.len(), 1, "Single row should produce 1 node");
    }

    #[test]
    fn test_build_single_table_power_of_two() {
        let table = create_random_table(4, 6);
        let tree = MultiTableTree::<F, Keccak256, 32>::build(&[table]).unwrap();
        // For height 4: 4 leaves + 2 nodes in next layer + 1 root = 7 total nodes = 2 * 4 - 1
        assert_eq!(tree.nodes.len(), 7, "Height 4 should produce 7 nodes (2 * 4 - 1)");
    }

    #[test]
    fn test_build_single_table_non_power_of_two() {
        let table = create_random_table(6, 4);
        let tree = MultiTableTree::<F, Keccak256, 32>::build(&[table]);
        
        assert!(matches!(
            tree,
            Err(MultiTableTreeError::WrongHeight),
        ));
    }

    #[test]
    fn test_build_valid_tables_different_heights() {
        let table_1 = create_random_table(2, 1);
        let table_2 = create_random_table(8, 2);
        let table_3 = create_random_table(4, 3);
        let table_4 = create_random_table(4, 4);
        
        let tree = MultiTableTree::<F, Keccak256, 32>::build(&[table_1, table_2, table_3, table_4]).unwrap();
        
        assert_eq!(tree.nodes.len(), 8 * 2 - 1);
    }

    #[test]
    fn test_build_invalid_tables_different_heights() {
        let table_1 = create_random_table(2, 3);
        let table_2 = create_random_table(8, 4);
        let table_3 = create_random_table(5, 2);
        
        let tree = MultiTableTree::<F, Keccak256, 32>::build(&[table_1, table_2, table_3]);
        
        assert!(matches!(
            tree,
            Err(MultiTableTreeError::WrongHeight),
        ));
    }

    #[test]
    fn test_build_deterministic() {
        let table_1 = create_random_table(4, 90);
        let table_2 = create_random_table(2, 100);
        
        let tree_1 = MultiTableTree::<F, Keccak256, 32>::build(&[table_1.clone(), table_2.clone()]).unwrap();
        let tree_2 = MultiTableTree::<F, Keccak256, 32>::build(&[table_1, table_2]).unwrap();
        
        assert_eq!(tree_1.root, tree_2.root, "Same input should produce same root");
        assert_eq!(tree_1.nodes, tree_2.nodes, "Same input should produce same nodes");
    }

    #[test]
    fn test_build_tables_order_doesnt_matter() {
        let table_1 = create_random_table(4, 90);
        let table_2 = create_random_table(2, 100);
        
        let tree_1 = MultiTableTree::<F, Keccak256, 32>::build(&[table_1.clone(), table_2.clone()]).unwrap();
        let tree_2 = MultiTableTree::<F, Keccak256, 32>::build(&[table_2, table_1]).unwrap();
        
        assert_eq!(tree_1.root, tree_2.root, "Same input in different order should produce same root");
        assert_eq!(tree_1.nodes, tree_2.nodes, "Same input in different order should produce same nodes");
    }

    #[test]
    fn compare_with_build_tree_by_hand() {
        let table_1 = create_random_table(4, 3);
        let table_2 = create_random_table(4, 2);
        let table_3 = create_random_table(2, 2);

        let tree = MultiTableTree::<F, Keccak256, 32>::build(&[table_1.clone(), table_2.clone(), table_3.clone()]).unwrap();

        let mut expected_nodes = Vec::new();

        // We build the first layer manually.
        for row_idx in 0..4 {
            let mut hasher = Keccak256::new();
            
            for element in table_1.get_row(row_idx).iter() {
                hasher.update(element.as_bytes());
            }
            for element in table_2.get_row(row_idx).iter() {
                hasher.update(element.as_bytes());
            }

            let mut result_hash = [0_u8; 32];
            result_hash.copy_from_slice(&hasher.finalize());
            expected_nodes.push(result_hash);
        }
        
        // Get the hashes of the row 0 of table_3:
        let mut hasher = Keccak256::new();
        for element in table_3.get_row(0) {
            hasher.update(element.as_bytes());
        } 
        let mut table_3_row_0_hash = [0_u8; 32];
        table_3_row_0_hash.copy_from_slice(&hasher.finalize());

        // Get the hashes of the row 1 of table_3:
        let mut hasher = Keccak256::new();
        for element in table_3.get_row(1) {
            hasher.update(element.as_bytes());
        } 
        let mut table_3_row_1_hash = [0_u8; 32];
        table_3_row_1_hash.copy_from_slice(&hasher.finalize());

        // Get first node of second layer:
        let mut hasher = Keccak256::new();
        hasher.update(expected_nodes[0]);
        hasher.update(expected_nodes[1]);
        hasher.update(table_3_row_0_hash);
        let mut first_node_second_layer_hash = [0_u8; 32];
        first_node_second_layer_hash.copy_from_slice(&hasher.finalize());
        expected_nodes.push(first_node_second_layer_hash);

        // Get second node of second layer:
        let mut hasher = Keccak256::new();
        hasher.update(expected_nodes[2]);
        hasher.update(expected_nodes[3]);
        hasher.update(table_3_row_1_hash);
        let mut second_node_second_layer_hash = [0_u8; 32];
        second_node_second_layer_hash.copy_from_slice(&hasher.finalize());
        expected_nodes.push(second_node_second_layer_hash);

        // Get root:
        let mut hasher = Keccak256::new();
        hasher.update(expected_nodes[4]);
        hasher.update(expected_nodes[5]);
        let mut root_node_hash = [0_u8; 32];
        root_node_hash.copy_from_slice(&hasher.finalize());
        expected_nodes.push(root_node_hash);
        
        // We define the expected tree:
        let expected_tree = MultiTableTree {
            root: expected_nodes[6].clone(),
            nodes: expected_nodes,
            injected_leaves: vec![
                vec![table_3_row_0_hash, table_3_row_1_hash],
            ],
            _phantom: PhantomData::<(F, Keccak256)>,
        };

        assert_eq!(expected_tree.nodes, tree.nodes);
        assert_eq!(expected_tree.injected_leaves, tree.injected_leaves);
    }

    #[test]
    fn test_injected_leaves_for_several_tables(){
        let table_1 = create_random_table(2, 1);
        let table_2 = create_random_table(8, 2);
        let table_3 = create_random_table(4, 3);
        let table_4 = create_random_table(4, 4);
        let table_5 = create_random_table(4, 5);
        let table_6 = create_random_table(2, 6);

        // We build the expected injected leaves:
        let mut expected_injected_leaves = Vec::new();

        let mut injected_length_4 = Vec::new();
        for row_idx in 0..4 {
            let mut hasher = Keccak256::new();
            
            for element in table_3.get_row(row_idx).iter() {
                hasher.update(element.as_bytes());
            }
            for element in table_4.get_row(row_idx).iter() {
                hasher.update(element.as_bytes());
            }
            for element in table_5.get_row(row_idx).iter() {
                hasher.update(element.as_bytes());
            }

            let mut result_hash = [0_u8; 32];
            result_hash.copy_from_slice(&hasher.finalize());
            injected_length_4.push(result_hash);
        }
        expected_injected_leaves.push(injected_length_4);
            
        let mut injected_length_2 = Vec::new();
        for row_idx in 0..2 {
            let mut hasher = Keccak256::new();
            
            for element in table_1.get_row(row_idx).iter() {
                hasher.update(element.as_bytes());
            }
            for element in table_6.get_row(row_idx).iter() {
                hasher.update(element.as_bytes());
            }

            let mut result_hash = [0_u8; 32];
            result_hash.copy_from_slice(&hasher.finalize());
            injected_length_2.push(result_hash);
        }
        expected_injected_leaves.push(injected_length_2);
        
        

        let tree = MultiTableTree::<F, Keccak256, 32>::build(&[
            table_1.clone(), 
            table_2.clone(), 
            table_3.clone(), 
            table_4.clone(), 
            table_5.clone(), 
            table_6.clone()]
        ).unwrap();

        assert_eq!(expected_injected_leaves, tree.injected_leaves);
    }
}