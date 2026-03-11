use alloc::vec::Vec;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

/// Sparse matrix in Compressed Sparse Row (CSR) format.
///
/// Used for the bipartite expander graphs in Brakedown's linear-time encodable code.
/// Each row stores the column indices of its nonzero entries and the corresponding
/// field-element values.
#[derive(Clone, Debug)]
pub struct SparseMatrix<F: IsField> {
    pub n_rows: usize,
    pub n_cols: usize,
    /// For each row: list of (column_index, value) pairs.
    entries: Vec<Vec<(usize, FieldElement<F>)>>,
}

impl<F: IsField> SparseMatrix<F> {
    /// Creates a new sparse matrix from explicit row entries.
    pub fn new(n_rows: usize, n_cols: usize, entries: Vec<Vec<(usize, FieldElement<F>)>>) -> Self {
        assert_eq!(entries.len(), n_rows);
        Self {
            n_rows,
            n_cols,
            entries,
        }
    }

    /// Multiplies this matrix by a column vector `v` of length `n_cols`,
    /// producing a vector of length `n_rows`.
    ///
    /// result\[i\] = sum_j A\[i\]\[j\] * v\[j\]
    pub fn mul_vec(&self, v: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        assert_eq!(v.len(), self.n_cols);
        self.entries
            .iter()
            .map(|row| {
                row.iter()
                    .fold(FieldElement::<F>::zero(), |acc, (col, val)| {
                        acc + val.clone() * v[*col].clone()
                    })
            })
            .collect()
    }

    /// Generates a random sparse matrix where each row has exactly `nnz_per_row`
    /// nonzero entries, each set to the field element `1`.
    ///
    /// Uses a simple deterministic approach based on the provided seed for
    /// reproducibility: for row i, the nonzero columns are chosen by a
    /// stride-based selection that covers distinct columns.
    pub fn random_binary(n_rows: usize, n_cols: usize, nnz_per_row: usize, seed: u64) -> Self {
        assert!(nnz_per_row <= n_cols);
        let one = FieldElement::<F>::one();
        let mut entries = Vec::with_capacity(n_rows);

        for i in 0..n_rows {
            let mut row = Vec::with_capacity(nnz_per_row);
            // Simple deterministic column selection using hash-like mixing
            let mut state = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64);
            let mut used = alloc::collections::BTreeSet::new();
            for _ in 0..nnz_per_row {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut col = (state >> 33) as usize % n_cols;
                while used.contains(&col) {
                    // Re-mix PRNG state instead of linear probing to avoid clustering bias
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    col = (state >> 33) as usize % n_cols;
                }
                used.insert(col);
                row.push((col, one.clone()));
            }
            entries.push(row);
        }

        Self {
            n_rows,
            n_cols,
            entries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    type F = U64PrimeField<101>;
    type FE = FieldElement<F>;

    #[test]
    fn mul_vec_basic() {
        // 2x3 sparse matrix with a few entries
        let entries = vec![
            vec![(0, FE::from(2)), (2, FE::from(3))], // row 0: 2*v[0] + 3*v[2]
            vec![(1, FE::from(5))],                   // row 1: 5*v[1]
        ];
        let m = SparseMatrix::new(2, 3, entries);
        let v = vec![FE::from(1), FE::from(2), FE::from(4)];
        let result = m.mul_vec(&v);
        assert_eq!(result[0], FE::from(14)); // 2*1 + 3*4
        assert_eq!(result[1], FE::from(10)); // 5*2
    }

    #[test]
    fn random_binary_structure() {
        let m = SparseMatrix::<F>::random_binary(4, 8, 3, 42);
        assert_eq!(m.n_rows, 4);
        assert_eq!(m.n_cols, 8);
        for row in &m.entries {
            assert_eq!(row.len(), 3);
            // Check all values are 1
            for (_, val) in row {
                assert_eq!(*val, FE::one());
            }
            // Check column indices are distinct
            let cols: Vec<usize> = row.iter().map(|(c, _)| *c).collect();
            let mut sorted = cols.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), cols.len());
        }
    }
}
