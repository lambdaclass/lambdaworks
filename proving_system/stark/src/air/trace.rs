use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::polynomial::Polynomial;

#[derive(Clone, Default, Debug)]
pub struct TraceTable<F: IsField> {
    /// `table` is column oriented trace element description
    pub table: Vec<FieldElement<F>>,
    pub n_cols: usize,
}

impl<F: IsField> TraceTable<F> {
    pub fn new_from_cols(cols: &[Vec<FieldElement<F>>]) -> Self {
        let n_cols = cols.len();
        let n_rows = cols[0].len();

        let mut table = Vec::with_capacity(n_cols * n_rows);

        for row_idx in 0..n_rows {
            for col in cols {
                table.push(col[row_idx].clone());
            }
        }
        Self { table, n_cols }
    }

    pub fn n_rows(&self) -> usize {
        self.table.len() / self.n_cols
    }

    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        let n_rows = self.n_rows();
        (0..n_rows)
            .map(|row_idx| {
                self.table[(row_idx * self.n_cols)..(row_idx * self.n_cols + self.n_cols)].to_vec()
            })
            .collect()
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.n_cols;
        &self.table[row_offset..row_offset + self.n_cols]
    }

    pub fn cols(&self) -> Vec<Vec<FieldElement<F>>> {
        let mut ret = Vec::with_capacity(self.n_cols);
        let n_rows = self.table.len() / self.n_cols;
        for row_idx in 0..n_rows {
            let mut col = Vec::with_capacity(n_rows);
            for col_idx in 0..self.n_cols {
                col.push(self.table[col_idx * self.n_cols + row_idx].clone())
            }
            ret.push(col);
        }
        ret
    }

    pub fn compute_trace_polys(
        &self,
        trace_roots_of_unity: &[FieldElement<F>],
    ) -> Vec<Polynomial<FieldElement<F>>> {
        self.cols()
            .iter()
            .map(|e| Polynomial::interpolate(&trace_roots_of_unity, e))
            .collect()
    }
}
