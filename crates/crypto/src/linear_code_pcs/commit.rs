use alloc::vec::Vec;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::merkle_tree::{merkle::MerkleTree, traits::IsMerkleTreeBackend};

use super::data_structures::{CommitOutput, CommitState, LinCodeCommitment};
use super::matrix::Matrix;
use super::traits::LinearCodeEncoding;

/// Commit to a multilinear polynomial using a linear-code PCS.
///
/// The polynomial is given as a flat vector of evaluations on the boolean hypercube
/// (length `2^v`). We arrange this into a `k x m` matrix, encode each row with
/// the provided linear code, and build a Merkle tree over the columns.
///
/// # Arguments
/// - `evals`: evaluation vector of length `2^v` (the multilinear polynomial in Lagrange basis).
/// - `encoding`: the linear code to use (RS or expander).
///
/// # Returns
/// A `CommitOutput` containing the public `LinCodeCommitment` and private `CommitState`.
///
/// # Type Parameters
/// - `B`: Merkle tree backend with `B::Data = Vec<FieldElement<F>>` (columns are leaves).
pub fn commit<F, B, E>(evals: &[FieldElement<F>], encoding: &E) -> CommitOutput<F, B>
where
    F: IsField,
    B: IsMerkleTreeBackend<Data = Vec<FieldElement<F>>>,
    E: LinearCodeEncoding<F>,
{
    let n = evals.len();
    assert!(n.is_power_of_two(), "evals length must be a power of two");
    let v = n.trailing_zeros() as usize;

    // Split variables: k rows, m cols where k * m = n
    let half = v / 2;
    let n_cols = 1usize << (v - half); // m = 2^ceil(v/2)
    let n_rows = 1usize << half; // k = 2^floor(v/2)

    assert_eq!(
        n_cols,
        encoding.message_len(),
        "encoding message_len must equal matrix column count"
    );

    // Arrange into k x m matrix (row-major)
    let matrix = Matrix::new(n_rows, n_cols, evals.to_vec());

    // Encode each row
    let n_ext_cols = encoding.codeword_len();
    let mut ext_data = Vec::with_capacity(n_rows * n_ext_cols);
    for i in 0..n_rows {
        let row = matrix.row(i);
        let encoded_row = encoding.encode(row);
        debug_assert_eq!(encoded_row.len(), n_ext_cols);
        ext_data.extend(encoded_row);
    }
    let ext_matrix = Matrix::new(n_rows, n_ext_cols, ext_data);

    // Build Merkle tree: each leaf is one column of the extended matrix
    let columns: Vec<Vec<FieldElement<F>>> = (0..n_ext_cols).map(|j| ext_matrix.col(j)).collect();

    let merkle_tree = MerkleTree::<B>::build(&columns)
        .expect("Merkle tree build should succeed for non-empty columns");

    let commitment = LinCodeCommitment {
        root: merkle_tree.root.clone(),
        n_rows,
        n_cols,
        n_ext_cols,
    };

    CommitOutput {
        commitment,
        state: CommitState {
            matrix,
            ext_matrix,
            merkle_tree,
        },
    }
}

/// Compute the tensor product vector: `tensor(values) = ⊗_{i} (1-r_i, r_i)`.
///
/// For `values = [r_0, r_1, ..., r_{k-1}]`, produces a vector of length `2^k`:
/// ```text
/// tensor[j] = prod_{i} ( if bit_i(j) then r_i else 1 - r_i )
/// ```
///
/// This is the key operation for the multilinear evaluation claim:
/// `f(r) = <tensor_L(r), M * tensor_R(r)> = tensor_L^T * M * tensor_R`.
pub fn tensor_vec<F: IsField>(values: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
    let n = values.len();
    let mut evals = alloc::vec![FieldElement::<F>::one(); 1 << n];
    let mut size = 1;
    for r in values {
        size *= 2;
        for i in (0..size).rev().step_by(2) {
            let scalar = evals[i / 2].clone();
            evals[i] = scalar.clone() * r.clone();
            evals[i - 1] = scalar - evals[i].clone();
        }
        // After processing variable r_k, index j has:
        // - bit (size/2) of j corresponds to r_0 (MSB = first variable)
        // - bit 1 of j corresponds to the most recently processed variable
        // This matches DenseMultilinearPolynomial's evaluation order.
    }
    evals
}
