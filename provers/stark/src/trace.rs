use crate::table::Table;
use lambdaworks_math::fft::errors::FFTError;
use lambdaworks_math::fft::polynomial::FFTPoly;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// A two-dimensional representation of an execution trace of the STARK
/// protocol.
///
/// For the moment it is mostly a wrapper around the `Table` struct. It is a
/// layer above the raw two-dimensional table, with functionality relevant to the
/// STARK protocol.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct TraceTable<F: IsFFTField> {
    pub table: Table<F>,
    pub step_size: usize,
}

impl<F: IsFFTField> TraceTable<F> {
    pub fn new(data: &[FieldElement<F>], n_columns: usize, step_size: usize) -> Self {
        let table = Table::new(data, n_columns);
        Self { table, step_size }
    }

    pub fn from_columns(columns: &[Vec<FieldElement<F>>], step_size: usize) -> Self {
        let table = Table::from_columns(columns);
        Self { table, step_size }
    }

    pub fn empty(step_size: usize) -> Self {
        Self::new(&Vec::new(), 0, step_size)
    }

    pub fn is_empty(&self) -> bool {
        self.table.width == 0
    }

    pub fn n_rows(&self) -> usize {
        self.table.height
    }

    pub fn num_steps(&self) -> usize {
        debug_assert!((self.table.height % self.step_size) == 0);
        self.table.height / self.step_size
    }

    pub fn n_cols(&self) -> usize {
        self.table.width
    }

    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        self.table.rows()
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        self.table.get_row(row_idx)
    }

    pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<F>] {
        self.table.get_row_mut(row_idx)
    }

    pub fn last_row(&self) -> &[FieldElement<F>] {
        self.get_row(self.n_rows() - 1)
    }

    pub fn columns(&self) -> Vec<Vec<FieldElement<F>>> {
        self.table.columns()
    }

    /// Given a slice of integer numbers representing column indexes, merge these columns into
    /// a one-dimensional vector.
    ///
    /// The particular way they are merged is not really important since this function is used to
    /// aggreagate values distributed across various columns with no importance on their ordering,
    /// such as to sort them.
    pub fn merge_columns(&self, column_indexes: &[usize]) -> Vec<FieldElement<F>> {
        let mut data = Vec::with_capacity(self.n_rows() * column_indexes.len());
        for row_index in 0..self.n_rows() {
            for column in column_indexes {
                data.push(self.table.data[row_index * self.n_cols() + column].clone());
            }
        }
        data
    }

    /// Given a row and a column index, gives stored value in that position
    pub fn get(&self, row: usize, col: usize) -> FieldElement<F> {
        self.table.get(row, col)
    }

    pub fn compute_trace_polys(&self) -> Vec<Polynomial<FieldElement<F>>>
    where
        FieldElement<F>: Send + Sync,
    {
        let columns = self.columns();
        #[cfg(feature = "parallel")]
        let iter = columns.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.iter();

        iter.map(|col| Polynomial::interpolate_fft(col))
            .collect::<Result<Vec<Polynomial<FieldElement<F>>>, FFTError>>()
            .unwrap()
    }

    pub fn concatenate(&self, new_cols: Vec<FieldElement<F>>, n_cols: usize) -> Self {
        let mut data = Vec::new();
        let mut i = 0;
        for row_index in (0..self.table.data.len()).step_by(self.table.width) {
            data.append(&mut self.table.data[row_index..row_index + self.table.width].to_vec());
            data.append(&mut new_cols[i..(i + n_cols)].to_vec());
            i += n_cols;
        }

        let table = Table::new(&data, self.n_cols() + n_cols);
        Self { table }
    }

    /// Given the padding length, appends the last row of the trace table
    /// that many times.
    /// This is useful for example when the desired trace length should be power
    /// of two, and only the last row is the one that can be appended without affecting
    /// the integrity of the constraints.
    pub fn pad_with_last_row(&mut self, padding_len: usize) {
        let last_row = self.last_row().to_vec();
        (0..padding_len).for_each(|_| {
            self.table.append_row(&last_row);
        })
    }

    /// Given a row index, a column index and a value, tries to set that location
    /// of the trace with the given value.
    /// The row_idx passed as argument may be greater than the max row index by 1. In this case,
    /// last row of the trace is cloned, and the value is set in that cloned row. Then, the row is
    /// appended to the end of the trace.
    pub fn set_or_extend(&mut self, row_idx: usize, col_idx: usize, value: &FieldElement<F>) {
        debug_assert!(col_idx < self.n_cols());
        // NOTE: This is not very nice, but for how the function is being used at the moment,
        // the passed `row_idx` should never be greater than `self.n_rows() + 1`. This is just
        // an integrity check for ease in the developing process, we should think a better alternative
        // in the future.
        debug_assert!(row_idx <= self.n_rows() + 1);
        if row_idx >= self.n_rows() {
            let mut last_row = self.last_row().to_vec();
            last_row[col_idx] = value.clone();
            self.table.append_row(&last_row)
        } else {
            let row = self.get_row_mut(row_idx);
            row[col_idx] = value.clone();
        }
    }
}

#[cfg(test)]
mod test {
    use super::TraceTable;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::F17};
    type FE = FieldElement<F17>;

    #[test]
    fn test_cols() {
        let col_1 = vec![FE::from(1), FE::from(2), FE::from(5), FE::from(13)];
        let col_2 = vec![FE::from(1), FE::from(3), FE::from(8), FE::from(21)];

        let trace_table = TraceTable::from_columns(&[col_1.clone(), col_2.clone()]);
        let res_cols = trace_table.columns();

        assert_eq!(res_cols, vec![col_1, col_2]);
    }

    #[test]
    fn test_concatenate_works() {
        let table1_columns = vec![vec![FE::new(7), FE::new(8), FE::new(9)]];
        let new_columns = vec![
            FE::new(1),
            FE::new(2),
            FE::new(3),
            FE::new(4),
            FE::new(5),
            FE::new(6),
        ];
        let expected_table = TraceTable::from_columns(&[
            vec![FE::new(7), FE::new(8), FE::new(9)],
            vec![FE::new(1), FE::new(3), FE::new(5)],
            vec![FE::new(2), FE::new(4), FE::new(6)],
        ]);
        let table1 = TraceTable::from_columns(&table1_columns);
        assert_eq!(table1.concatenate(new_columns, 2), expected_table)
    }
}
