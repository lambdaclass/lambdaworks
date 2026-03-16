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
    /// Uses SHA3-256 to derive an independent PRNG seed per row from (seed, row_index),
    /// ensuring each row's column selection is pseudorandom and uncorrelated.
    pub fn random_binary(n_rows: usize, n_cols: usize, nnz_per_row: usize, seed: u64) -> Self {
        use sha3::{Digest, Sha3_256};

        /// Extract the first 8 bytes of a 32-byte hash as a u64.
        fn u64_from_hash(hash: &[u8; 32]) -> u64 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&hash[..8]);
            u64::from_le_bytes(buf)
        }

        /// Hash (seed, row, counter) into a 32-byte digest.
        fn derive(seed: u64, row: u64, counter: u64) -> [u8; 32] {
            let mut h = Sha3_256::new();
            h.update(seed.to_le_bytes());
            h.update(row.to_le_bytes());
            h.update(counter.to_le_bytes());
            h.finalize().into()
        }

        assert!(nnz_per_row <= n_cols);
        let one = FieldElement::<F>::one();
        let mut entries = Vec::with_capacity(n_rows);

        for i in 0..n_rows {
            let mut row = Vec::with_capacity(nnz_per_row);
            let row_idx = i as u64;

            // Derive initial per-row state via SHA3-256(seed || row_index || 0)
            let hash = derive(seed, row_idx, 0);
            let mut state = u64_from_hash(&hash);
            let mut hash_counter: u64 = 1;
            let mut used = alloc::collections::BTreeSet::new();

            for _ in 0..nnz_per_row {
                let mut col = (state >> 1) as usize % n_cols;
                while used.contains(&col) {
                    let rehash = derive(seed, row_idx, hash_counter);
                    state = u64_from_hash(&rehash);
                    hash_counter += 1;
                    col = (state >> 1) as usize % n_cols;
                }
                used.insert(col);
                row.push((col, one.clone()));

                // Advance state for next column pick
                let next_hash = derive(seed, row_idx, hash_counter);
                state = u64_from_hash(&next_hash);
                hash_counter += 1;
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
    fn consecutive_rows_have_different_column_patterns() {
        let m = SparseMatrix::<F>::random_binary(100, 64, 3, 42);
        let mut identical_count = 0;
        for i in 0..m.n_rows - 1 {
            let mut cols_i: Vec<usize> = m.entries[i].iter().map(|(c, _)| *c).collect();
            let mut cols_next: Vec<usize> = m.entries[i + 1].iter().map(|(c, _)| *c).collect();
            cols_i.sort();
            cols_next.sort();
            if cols_i == cols_next {
                identical_count += 1;
            }
        }
        assert_eq!(
            identical_count, 0,
            "consecutive rows should not share identical column sets"
        );
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
