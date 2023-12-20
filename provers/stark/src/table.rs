use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

use crate::trace::StepView;

/// A two-dimensional Table holding field elements, arranged in a row-major order.
/// This is the basic underlying data structure used for any two-dimensional component in the
/// the STARK protocol implementation, such as the `TraceTable` and the `EvaluationFrame`.
/// Since this struct is a representation of a two-dimensional table, all rows should have the same
/// length.
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Table<F: IsField> {
    pub data: Vec<FieldElement<F>>,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EvaluationTable<F: IsSubFieldOf<E>, E: IsField> {
    pub(crate) main_table: Table<F>,
    pub(crate) aux_table: Table<E>,
    pub(crate) step_size: usize,
}

impl<'t, F: IsField> Table<F> {
    /// Crates a new Table instance from a one-dimensional array in row major order
    /// and the intended width of the table.
    pub fn new(data: Vec<FieldElement<F>>, width: usize) -> Self {
        // Check if the intented width is 0, used for creating an empty table.
        if width == 0 {
            return Self {
                data: Vec::new(),
                width,
                height: 0,
            };
        }

        // Check that the one-dimensional data makes sense to be interpreted as a 2D one.
        debug_assert!(crate::debug::validate_2d_structure(&data, width));
        let height = data.len() / width;

        Self {
            data,
            width,
            height,
        }
    }

    /// Creates a Table instance from a vector of the intended columns.
    pub fn from_columns(columns: Vec<Vec<FieldElement<F>>>) -> Self {
        if columns.is_empty() {
            return Self::new(Vec::new(), 0);
        }
        let height = columns[0].len();

        // Check that all columns have the same length for integrity
        debug_assert!(columns.iter().all(|c| c.len() == height));

        let width = columns.len();
        let mut data = Vec::with_capacity(width * height);

        for row_idx in 0..height {
            for column in columns.iter() {
                data.push(column[row_idx].clone());
            }
        }

        Self::new(data, width)
    }

    /// Returns a vector of vectors of field elements representing the table rows
    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        self.data.chunks(self.width).map(|r| r.to_vec()).collect()
    }

    /// Given a row index, returns a reference to that row as a slice of field elements.
    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.width;
        &self.data[row_offset..row_offset + self.width]
    }

    /// Given a row index, returns a mutable reference to that row as a slice of field elements.
    pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<F>] {
        let n_cols = self.width;
        let row_offset = row_idx * n_cols;
        &mut self.data[row_offset..row_offset + n_cols]
    }

    /// Given a row index and a number of rows, returns a view of a subset of contiguous rows
    /// of the table, starting from that index.
    pub fn table_view(&'t self, from_idx: usize, num_rows: usize) -> TableView<'t, F> {
        let from_offset = from_idx * self.width;
        let data = &self.data[from_offset..from_offset + self.width * num_rows];

        TableView {
            data,
            table_row_idx: from_idx,
            width: self.width,
            height: num_rows,
        }
    }

    /// Given a slice of field elements representing a row, appends it to
    /// the end of the table.
    pub fn append_row(&mut self, row: &[FieldElement<F>]) {
        debug_assert_eq!(row.len(), self.width);
        self.data.extend_from_slice(row);
        self.height += 1
    }

    /// Returns a reference to the last row of the table
    pub fn last_row(&self) -> &[FieldElement<F>] {
        self.get_row(self.height - 1)
    }

    /// Returns a vector of vectors of field elements representing the table
    /// columns
    pub fn columns(&self) -> Vec<Vec<FieldElement<F>>> {
        (0..self.width)
            .map(|col_idx| {
                (0..self.height)
                    .map(|row_idx| self.data[row_idx * self.width + col_idx].clone())
                    .collect()
            })
            .collect()
    }

    /// Given row and column indexes, returns the stored field element in that position of the table.
    pub fn get(&self, row: usize, col: usize) -> &FieldElement<F> {
        let idx = row * self.width + col;
        &self.data[idx]
    }
}

impl<E: IsField> EvaluationTable<E, E> {
    pub fn get_row(&self, row_idx: usize) -> Vec<FieldElement<E>> {
        let mut row: Vec<_> = self.get_row_main(row_idx).to_vec();
        row.extend_from_slice(self.get_row_aux(row_idx));
        row
    }

    pub fn columns(&self) -> Vec<Vec<FieldElement<E>>> {
        let mut columns = self.main_table.columns();
        let aux_columns = self.aux_table.columns();
        columns.extend(aux_columns);
        columns
    }
}

impl<F: IsSubFieldOf<E>, E: IsField> EvaluationTable<F, E> {
    /// Creates a Table instance from a vector of the intended columns.
    pub fn from_columns(
        main_columns: Vec<Vec<FieldElement<F>>>,
        aux_columns: Vec<Vec<FieldElement<E>>>,
        step_size: usize,
    ) -> Self {
        Self {
            main_table: Table::from_columns(main_columns),
            aux_table: Table::from_columns(aux_columns),
            step_size,
        }
    }

    pub fn n_main_cols(&self) -> usize {
        self.main_table.width
    }

    pub fn n_aux_cols(&self) -> usize {
        self.aux_table.width
    }

    pub fn n_cols(&self) -> usize {
        self.n_main_cols() + self.n_aux_cols()
    }

    pub fn num_steps(&self) -> usize {
        debug_assert!((self.main_table.height % self.step_size) == 0);
        debug_assert!(
            self.aux_table.height == 0 || (self.main_table.height == self.aux_table.height)
        );
        self.main_table.height / self.step_size
    }

    /// Given a particular step of the computation represented on the trace,
    /// returns the row of the underlying table.
    pub fn step_to_row(&self, step: usize) -> usize {
        self.step_size * step
    }

    /// Given a step index, return the step view of the trace for that index
    pub fn step_view<'t>(&'t self, step_idx: usize) -> StepView<'t, F, E> {
        let row_idx = self.step_to_row(step_idx);
        let main_table_view = self.main_table.table_view(row_idx, self.step_size);
        let aux_table_view = self.aux_table.table_view(row_idx, self.step_size);

        StepView {
            main_table_view,
            aux_table_view,
            step_idx,
        }
    }

    /// Given a row and a column index, gives stored value in that position
    pub fn get_main(&self, row: usize, col: usize) -> &FieldElement<F> {
        self.main_table.get(row, col)
    }

    /// Given a row and a column index, gives stored value in that position
    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
        self.aux_table.get(row, col)
    }

    /// Given a row index, returns a reference to that row in the main table as a slice of field elements.
    pub fn get_row_main(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.main_table.width;
        &self.main_table.data[row_offset..row_offset + self.main_table.width]
    }

    /// Given a row index, returns a reference to that row in the aux table as a slice of field elements.
    pub fn get_row_aux(&self, row_idx: usize) -> &[FieldElement<E>] {
        let row_offset = row_idx * self.aux_table.width;
        &self.aux_table.data[row_offset..row_offset + self.aux_table.width]
    }

    pub fn n_rows(&self) -> usize {
        debug_assert!(
            self.aux_table.height == 0 || (self.main_table.height == self.aux_table.height)
        );
        self.main_table.height
    }
}

/// A view of a contiguos subset of rows of a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableView<'t, F: IsField> {
    pub data: &'t [FieldElement<F>],
    pub table_row_idx: usize,
    pub width: usize,
    pub height: usize,
}

impl<'t, F: IsField> TableView<'t, F> {
    pub fn new(
        data: &'t [FieldElement<F>],
        table_row_idx: usize,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            data,
            width,
            table_row_idx,
            height,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> &FieldElement<F> {
        let idx = row * self.width + col;
        &self.data[idx]
    }

    pub fn get_row(&self, row: usize) -> &[FieldElement<F>] {
        let first = row * self.width;
        &self.data[first..first + self.width]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Felt252;

    #[test]
    fn get_rows_slice_works() {
        let data: Vec<Felt252> = (0..=11).map(Felt252::from).collect();
        let table = Table::new(data, 3);

        let slice = table.table_view(1, 2);
        let expected_data: Vec<Felt252> = (3..=8).map(Felt252::from).collect();

        assert_eq!(slice.data, expected_data);
    }
}
