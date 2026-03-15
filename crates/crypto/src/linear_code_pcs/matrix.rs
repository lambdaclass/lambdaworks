use alloc::vec::Vec;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

/// Row-major dense matrix of field elements.
///
/// Used to arrange polynomial evaluations for the Ligero/Brakedown commit phase:
/// the evaluation vector is reshaped into a `n_rows x n_cols` matrix, each row is
/// encoded with a linear code, and columns of the extended matrix become Merkle leaves.
#[derive(Clone, Debug)]
pub struct Matrix<F: IsField> {
    pub n_rows: usize,
    pub n_cols: usize,
    data: Vec<FieldElement<F>>,
}

impl<F: IsField> Matrix<F> {
    /// Build a matrix from a flat vector of field elements in row-major order.
    ///
    /// # Panics
    /// Panics if `data.len() != n_rows * n_cols`.
    pub fn new(n_rows: usize, n_cols: usize, data: Vec<FieldElement<F>>) -> Self {
        assert_eq!(
            data.len(),
            n_rows * n_cols,
            "data length {} != n_rows * n_cols = {}",
            data.len(),
            n_rows * n_cols
        );
        Self {
            n_rows,
            n_cols,
            data,
        }
    }

    /// Returns a slice representing row `i`.
    pub fn row(&self, i: usize) -> &[FieldElement<F>] {
        let start = i * self.n_cols;
        &self.data[start..start + self.n_cols]
    }

    /// Returns column `j` as a newly allocated vector.
    pub fn col(&self, j: usize) -> Vec<FieldElement<F>> {
        (0..self.n_rows)
            .map(|i| self.data[i * self.n_cols + j].clone())
            .collect()
    }

    /// Left-multiply by a row vector `v` of length `n_rows`:
    /// result\[j\] = sum_i v\[i\] * M\[i\]\[j\]  (length `n_cols`).
    ///
    /// This computes `v^T * M`, producing a vector of length `n_cols`.
    pub fn row_mul(&self, v: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        assert_eq!(v.len(), self.n_rows);
        let mut result = vec![FieldElement::<F>::zero(); self.n_cols];
        for (i, vi) in v.iter().enumerate() {
            let row = self.row(i);
            for (rj, res_j) in row.iter().zip(result.iter_mut()) {
                *res_j = res_j.clone() + vi.clone() * rj.clone();
            }
        }
        result
    }

    /// Returns a reference to the underlying flat data.
    pub fn data(&self) -> &[FieldElement<F>] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    type F = U64PrimeField<101>;
    type FE = FieldElement<F>;

    #[test]
    fn row_col_access() {
        // 2x3 matrix: [[1,2,3],[4,5,6]]
        let data: Vec<FE> = (1..=6).map(|x| FE::from(x as u64)).collect();
        let m = Matrix::new(2, 3, data);

        assert_eq!(m.row(0), &[FE::from(1), FE::from(2), FE::from(3)]);
        assert_eq!(m.row(1), &[FE::from(4), FE::from(5), FE::from(6)]);

        assert_eq!(m.col(0), vec![FE::from(1), FE::from(4)]);
        assert_eq!(m.col(1), vec![FE::from(2), FE::from(5)]);
        assert_eq!(m.col(2), vec![FE::from(3), FE::from(6)]);
    }

    #[test]
    fn row_mul_identity_like() {
        // 2x2 matrix: [[1,0],[0,1]], v = [3,5]
        let m = Matrix::new(
            2,
            2,
            vec![FE::from(1), FE::from(0), FE::from(0), FE::from(1)],
        );
        let v = vec![FE::from(3), FE::from(5)];
        let result = m.row_mul(&v);
        assert_eq!(result, vec![FE::from(3), FE::from(5)]);
    }

    #[test]
    fn row_mul_general() {
        // 2x3 matrix: [[1,2,3],[4,5,6]], v = [2,3]
        // result[0] = 2*1 + 3*4 = 14
        // result[1] = 2*2 + 3*5 = 19
        // result[2] = 2*3 + 3*6 = 24
        let data: Vec<FE> = (1..=6).map(|x| FE::from(x as u64)).collect();
        let m = Matrix::new(2, 3, data);
        let v = vec![FE::from(2), FE::from(3)];
        let result = m.row_mul(&v);
        assert_eq!(result, vec![FE::from(14), FE::from(19), FE::from(24)]);
    }
}
