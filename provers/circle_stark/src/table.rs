use lambdaworks_math::field::{
    element::FieldElement, fields::mersenne31::field::Mersenne31Field,
};

/// A two-dimensional Table holding field elements, arranged in a row-major order.
/// This is the basic underlying data structure used for any two-dimensional component in the
/// the STARK protocol implementation, such as the `TraceTable` and the `EvaluationFrame`.
/// Since this struct is a representation of a two-dimensional table, all rows should have the same
/// length.
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Table {
    pub data: Vec<FieldElement<Mersenne31Field>>,
    pub width: usize,
    pub height: usize,
}

impl Table {
    /// Crates a new Table instance from a one-dimensional array in row major order
    /// and the intended width of the table.
    pub fn new(data: Vec<FieldElement<Mersenne31Field>>, width: usize) -> Self {
        // Check if the intented width is 0, used for creating an empty table.
        if width == 0 {
            return Self {
                data: Vec::new(),
                width,
                height: 0,
            };
        }

        // Check that the one-dimensional data makes sense to be interpreted as a 2D one.
        // debug_assert!(crate::debug::validate_2d_structure(&data, width));
        let height = data.len() / width;

        Self {
            data,
            width,
            height,
        }
    }

    /// Creates a Table instance from a vector of the intended columns.
    pub fn from_columns(columns: Vec<Vec<FieldElement<Mersenne31Field>>>) -> Self {
        if columns.is_empty() {
            return Self::new(Vec::new(), 0);
        }
        let height = columns[0].len();

        // Check that all columns have the same length for integrity
        // debug_assert!(columns.iter().all(|c| c.len() == height));

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
    pub fn rows(&self) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
        self.data.chunks(self.width).map(|r| r.to_vec()).collect()
    }

    /// Given a row index, returns a reference to that row as a slice of field elements.
    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<Mersenne31Field>] {
        let row_offset = row_idx * self.width;
        &self.data[row_offset..row_offset + self.width]
    }

    /// Given a row index, returns a mutable reference to that row as a slice of field elements.
    pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<Mersenne31Field>] {
        let n_cols = self.width;
        let row_offset = row_idx * n_cols;
        &mut self.data[row_offset..row_offset + n_cols]
    }

    /// Given a slice of field elements representing a row, appends it to
    /// the end of the table.
    pub fn append_row(&mut self, row: &[FieldElement<Mersenne31Field>]) {
        debug_assert_eq!(row.len(), self.width);
        self.data.extend_from_slice(row);
        self.height += 1
    }

    /// Returns a reference to the last row of the table
    pub fn last_row(&self) -> &[FieldElement<Mersenne31Field>] {
        self.get_row(self.height - 1)
    }

    /// Returns a vector of vectors of field elements representing the table
    /// columns
    pub fn columns(&self) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
        (0..self.width)
            .map(|col_idx| {
                (0..self.height)
                    .map(|row_idx| self.data[row_idx * self.width + col_idx].clone())
                    .collect()
            })
            .collect()
    }

    /// Given row and column indexes, returns the stored field element in that position of the table.
    pub fn get(&self, row: usize, col: usize) -> &FieldElement<Mersenne31Field> {
        let idx = row * self.width + col;
        &self.data[idx]
    }
}
