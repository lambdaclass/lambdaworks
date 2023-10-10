use crate::table::Table;
use lambdaworks_math::fft::errors::FFTError;
use lambdaworks_math::fft::polynomial::FFTPoly;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct TraceTable<F: IsFFTField> {
    pub table: Table<F>,
}

impl<F: IsFFTField> TraceTable<F> {
    pub fn new(data: &[FieldElement<F>], n_columns: usize) -> Self {
        let table = Table::new(data, n_columns);
        Self { table }
    }

    pub fn from_columns(columns: &[Vec<FieldElement<F>>]) -> Self {
        let table = Table::from_columns(columns);
        Self { table }
    }

    pub fn empty() -> Self {
        Self::new(&Vec::new(), 0)
    }

    pub fn is_empty(&self) -> bool {
        self.table.width == 0
    }

    pub fn n_rows(&self) -> usize {
        self.table.height
    }

    pub fn n_cols(&self) -> usize {
        self.table.width
    }

    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        self.table.rows()
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.n_cols();
        &self.table.data[row_offset..row_offset + self.n_cols()]
    }

    pub fn last_row(&self) -> &[FieldElement<F>] {
        self.get_row(self.n_rows() - 1)
    }

    pub fn columns(&self) -> Vec<Vec<FieldElement<F>>> {
        self.table.columns()
    }

    pub fn get_columns(&self, columns: &[usize]) -> Vec<FieldElement<F>> {
        self.table.get_columns(columns)
    }

    /// Given a row and a column index, gives stored value in that position
    pub fn get(&self, row: usize, col: usize) -> FieldElement<F> {
        self.table.get(row, col)
    }

    pub fn compute_trace_polys(&self) -> Vec<Polynomial<FieldElement<F>>> {
        self.columns()
            .iter()
            .map(|col| Polynomial::interpolate_fft(col))
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

    pub fn pad_with_last_row(&mut self, number_rows: usize) {
        let last_row = self.last_row().to_vec();
        (0..number_rows).for_each(|_| {
            self.table.append_row(&last_row);
        })
    }

    pub fn add_to_column(&mut self, i: usize, value: &FieldElement<F>, col: usize) {
        let trace_idx = i * self.n_cols() + col;
        if trace_idx >= self.table.data.len() {
            let mut last_row = self.last_row().to_vec();
            last_row[col] = value.clone();
            self.table.append_row(&last_row)
        } else {
            self.table.data[trace_idx] = value.clone();
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
