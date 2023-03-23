use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::polynomial::Polynomial;

#[derive(Clone, Default, Debug)]
pub struct TraceTable<F: IsField> {
    /// `table` is column oriented trace element description
    pub table: Vec<Vec<FieldElement<F>>>,
}

impl<F: IsField> TraceTable<F> {
    pub fn rows(&self) -> Vec<Vec<&FieldElement<F>>> {
        // All columns should be of the same size, so we take the
        // len of the first one to know the number of rows.
        let n_rows = self.table[0].len();
        let n_cols = self.table.len();
        let ret = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let row = Vec::with_capacity(n_cols);
            for col_idx in 0..n_cols {
                row.push(&self.table[col_idx][row_idx]);
            }
            ret.push(row);
        }
        ret
    }

    pub fn get_row(&self, row_idx: usize) -> Vec<&FieldElement<F>> {
        let n_cols = self.table.len();
        let mut ret = Vec::with_capacity(n_cols);
        (0..n_cols).for_each(|col_idx| ret.push(&self.table[col_idx][row_idx]));
        ret
    }

    pub fn cols(&self) -> &[Vec<FieldElement<F>>] {
        &self.table
    }

    pub fn compute_trace_polys(
        &self,
        trace_roots_of_unity: &[FieldElement<F>],
    ) -> Vec<Polynomial<FieldElement<F>>> {
        self.table
            .iter()
            .map(|e| Polynomial::interpolate(&trace_roots_of_unity, e))
            .collect()
    }
}
