use super::trace::TraceTable;
use crate::table::Table;
use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};

#[derive(Clone, Debug, PartialEq)]
pub struct Frame<F: IsFFTField> {
    table: Table<F>,
}

impl<F: IsFFTField> Frame<F> {
    pub fn new(data: Vec<FieldElement<F>>, row_width: usize) -> Self {
        let table = Table::new(data, row_width);
        Self { table }
    }

    pub fn n_rows(&self) -> usize {
        self.table.height
    }

    pub fn n_cols(&self) -> usize {
        self.table.width
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        self.table.get_row(row_idx)
    }

    pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<F>] {
        self.table.get_row_mut(row_idx)
    }

    pub fn read_from_trace(
        trace: &TraceTable<F>,
        step: usize,
        blowup: u8,
        offsets: &[usize],
    ) -> Self {
        // Get trace length to apply module with it when getting elements of
        // the frame from the trace.
        let trace_steps = trace.n_rows();
        let data = offsets
            .iter()
            .flat_map(|frame_row_idx| {
                trace
                    .get_row((step + (frame_row_idx * blowup as usize)) % trace_steps)
                    .to_vec()
            })
            .collect();

        Self::new(data, trace.table.width)
    }
}

impl<F: IsFFTField> From<&Table<F>> for Frame<F> {
    fn from(value: &Table<F>) -> Self {
        Self::new(value.data.clone(), value.width)
    }
}
