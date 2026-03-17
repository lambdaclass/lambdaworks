//! Sparse matrix in COO (Coordinate) format for R1CS.

use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::errors::SpartanError;

/// A single non-zero entry in a sparse matrix.
#[derive(Clone, Debug)]
pub struct SparseEntry<F: IsField>
where
    F::BaseType: Send + Sync,
{
    pub row: usize,
    pub col: usize,
    pub val: FieldElement<F>,
}

/// Sparse matrix in COO (Coordinate) format, sorted by (row, col).
#[derive(Clone, Debug)]
pub struct SparseMatrix<F: IsField>
where
    F::BaseType: Send + Sync,
{
    pub entries: Vec<SparseEntry<F>>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl<F: IsField> SparseMatrix<F>
where
    F::BaseType: Send + Sync,
{
    /// Creates a new sparse matrix from COO entries.
    ///
    /// Filters out zero entries, sorts by (row, col), and validates bounds.
    pub fn new(
        entries: Vec<SparseEntry<F>>,
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Self, SpartanError> {
        let mut filtered: Vec<SparseEntry<F>> = entries
            .into_iter()
            .filter(|e| e.val != FieldElement::zero())
            .collect();

        for e in &filtered {
            if e.row >= num_rows || e.col >= num_cols {
                return Err(SpartanError::R1CSError(format!(
                    "sparse entry ({}, {}) out of bounds for {}x{} matrix",
                    e.row, e.col, num_rows, num_cols
                )));
            }
        }

        filtered.sort_by(|a, b| a.row.cmp(&b.row).then(a.col.cmp(&b.col)));

        Ok(Self {
            entries: filtered,
            num_rows,
            num_cols,
        })
    }

    /// Creates a sparse matrix from a dense `Vec<Vec<FieldElement<F>>>`.
    pub fn from_dense(dense: &[Vec<FieldElement<F>>]) -> Self {
        let num_rows = dense.len();
        let num_cols = if num_rows > 0 { dense[0].len() } else { 0 };
        let mut entries = Vec::new();

        for (i, row) in dense.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                if *val != FieldElement::zero() {
                    entries.push(SparseEntry {
                        row: i,
                        col: j,
                        val: val.clone(),
                    });
                }
            }
        }
        // Already sorted by (row, col) due to iteration order.
        Self {
            entries,
            num_rows,
            num_cols,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_from_dense_extracts_nonzeros() {
        let z = FE::zero();
        let one = FE::one();
        let two = FE::from(2u64);
        let dense = vec![
            vec![one.clone(), z.clone(), two.clone()],
            vec![z.clone(), z.clone(), one.clone()],
        ];
        let sparse = SparseMatrix::from_dense(&dense);
        assert_eq!(sparse.num_rows, 2);
        assert_eq!(sparse.num_cols, 3);
        assert_eq!(sparse.entries.len(), 3); // (0,0,1), (0,2,2), (1,2,1)
        assert_eq!(sparse.entries[0].row, 0);
        assert_eq!(sparse.entries[0].col, 0);
        assert_eq!(sparse.entries[1].row, 0);
        assert_eq!(sparse.entries[1].col, 2);
        assert_eq!(sparse.entries[2].row, 1);
        assert_eq!(sparse.entries[2].col, 2);
    }

    #[test]
    fn test_from_dense_all_zeros() {
        let z = FE::zero();
        let dense = vec![vec![z.clone(), z.clone()], vec![z.clone(), z.clone()]];
        let sparse = SparseMatrix::from_dense(&dense);
        assert!(sparse.entries.is_empty());
    }

    #[test]
    fn test_new_filters_zeros_and_sorts() {
        let one = FE::one();
        let two = FE::from(2u64);
        let entries = vec![
            SparseEntry {
                row: 1,
                col: 0,
                val: two,
            },
            SparseEntry {
                row: 0,
                col: 1,
                val: FE::zero(),
            }, // should be filtered
            SparseEntry {
                row: 0,
                col: 0,
                val: one,
            },
        ];
        let m = SparseMatrix::new(entries, 2, 2).unwrap();
        assert_eq!(m.entries.len(), 2);
        assert_eq!(m.entries[0].row, 0); // sorted first
        assert_eq!(m.entries[1].row, 1);
    }

    #[test]
    fn test_new_validates_bounds() {
        let one = FE::one();
        let entries = vec![SparseEntry {
            row: 5,
            col: 0,
            val: one,
        }];
        assert!(SparseMatrix::new(entries, 2, 2).is_err());
    }
}
