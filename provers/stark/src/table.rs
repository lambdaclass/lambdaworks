use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};

/// A two-dimensional Table holding field elements, arranged in a row-major order.
/// This is the basic underlying data structure used for any two-dimensional component in the
/// the STARK protocol implementation, such as the `TraceTable` and the `EvaluationFrame`.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Table<F: IsFFTField> {
    pub data: Vec<FieldElement<F>>,
    pub width: usize,
    pub height: usize,
}

impl<F: IsFFTField> Table<F> {
    pub fn new(data: &[FieldElement<F>], width: usize) -> Self {
        if width == 0 {
            return Self {
                data: Vec::new(),
                width,
                height: 0,
            };
        }

        debug_assert!((|d: &[FieldElement<F>]| {
            let rows: Vec<Vec<FieldElement<F>>> = d.chunks(width).map(|c| c.to_vec()).collect();
            rows.iter().all(|r| r.len() == rows[0].len())
        })(data));

        let height = data.len() / width;
        Self {
            data: data.to_vec(),
            width,
            height,
        }
    }

    pub fn from_columns(columns: &[Vec<FieldElement<F>>]) -> Self {
        if columns.is_empty() {
            return Self::new(&Vec::new(), 0);
        }
        let height = columns[0].len();
        debug_assert!(columns.iter().all(|c| c.len() == height));

        let width = columns.len();
        let mut data = Vec::with_capacity(width * height);
        for row_idx in 0..height {
            for column in columns {
                data.push(column[row_idx].clone());
            }
        }

        Self::new(&data, width)
    }

    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        (0..self.height)
            .map(|row_idx| {
                self.data[(row_idx * self.width)..(row_idx * self.width + self.width)].to_vec()
            })
            .collect()
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.width;
        &self.data[row_offset..row_offset + self.width]
    }

    pub fn append_row(&mut self, row: &[FieldElement<F>]) {
        self.data.extend_from_slice(row);
        self.height += 1
    }

    pub fn last_row(&self) -> &[FieldElement<F>] {
        self.get_row(self.height - 1)
    }

    pub fn columns(&self) -> Vec<Vec<FieldElement<F>>> {
        (0..self.width)
            .map(|col_idx| {
                (0..self.height)
                    .map(|row_idx| self.data[row_idx * self.width + col_idx].clone())
                    .collect()
            })
            .collect()
    }

    pub fn get_columns(&self, columns: &[usize]) -> Vec<FieldElement<F>> {
        let mut data = Vec::with_capacity(self.height * columns.len());
        for row_index in 0..self.height {
            for column in columns {
                data.push(self.data[row_index * self.width + column].clone());
            }
        }
        data
    }

    /// Given row and column indexes, gives stored value in that position of the table.
    pub fn get(&self, row: usize, col: usize) -> FieldElement<F> {
        let idx = row * self.width + col;
        self.data[idx].clone()
    }
}
