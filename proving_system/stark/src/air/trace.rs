use lambdaworks_fft::errors::FFTError;
use lambdaworks_fft::polynomial::FFTPoly;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

use super::errors::AIRError;

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct TraceTable<F: IsFFTField> {
    /// `table` is row-major trace element description
    pub table: Vec<FieldElement<F>>,
    pub n_cols: usize,
}

impl<F: IsFFTField> TraceTable<F> {
    pub fn empty() -> Self {
        Self {
            table: Vec::new(),
            n_cols: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.n_cols == 0
    }

    pub fn new_from_cols(cols: &[Vec<FieldElement<F>>]) -> Self {
        let n_rows = cols[0].len();
        debug_assert!(cols.iter().all(|c| c.len() == n_rows));

        let n_cols = cols.len();

        let mut table = Vec::with_capacity(n_cols * n_rows);

        for row_idx in 0..n_rows {
            for col in cols {
                table.push(col[row_idx].clone());
            }
        }
        Self { table, n_cols }
    }

    pub fn n_rows(&self) -> usize {
        if self.n_cols == 0 {
            0
        } else {
            self.table.len() / self.n_cols
        }
    }

    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        let n_rows = self.n_rows();
        (0..n_rows)
            .map(|row_idx| {
                self.table[(row_idx * self.n_cols)..(row_idx * self.n_cols + self.n_cols)].to_vec()
            })
            .collect()
    }

    pub fn get_row(&self, row_idx: usize) -> Result<&[FieldElement<F>], AIRError> {
        let row_offset = row_idx * self.n_cols;
        self.table
            .get(row_offset..row_offset + self.n_cols)
            .ok_or(AIRError::RowIndexOutOfTableBounds(row_idx, self.n_rows()))
    }

    pub fn cols(&self) -> Vec<Vec<FieldElement<F>>> {
        let n_rows = self.n_rows();
        (0..self.n_cols)
            .map(|col_idx| {
                (0..n_rows)
                    .map(|row_idx| self.table[row_idx * self.n_cols + col_idx].clone())
                    .collect()
            })
            .collect()
    }

    /// Given a step and a column index, gives stored value in that position
    pub fn get(&self, step: usize, col: usize) -> FieldElement<F> {
        let idx = step * self.n_cols + col;
        self.table[idx].clone()
    }

    pub fn compute_trace_polys(&self) -> Result<Vec<Polynomial<FieldElement<F>>>, AIRError> {
        self.cols()
            .iter()
            .map(|col| Polynomial::interpolate_fft(col))
            .collect::<Result<Vec<Polynomial<FieldElement<F>>>, FFTError>>()
            .map_err(AIRError::TracePolynomialsComputation)
    }
}

#[cfg(test)]
mod test {
    use super::TraceTable;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::F17};

    #[test]
    fn test_cols() {
        type F = FieldElement<F17>;

        let col_1 = vec![F::from(1), F::from(2), F::from(5), F::from(13)];
        let col_2 = vec![F::from(1), F::from(3), F::from(8), F::from(21)];

        let trace_table = TraceTable::new_from_cols(&[col_1.clone(), col_2.clone()]);
        let res_cols = trace_table.cols();

        assert_eq!(res_cols, vec![col_1, col_2]);
    }
}
