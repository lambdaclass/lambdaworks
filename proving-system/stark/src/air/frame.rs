use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use super::trace::TraceTable;

// Hard-coded for fibonacci
pub struct Frame<F: IsField> {
    data: Vec<FieldElement<F>>,
    row_width: usize,
}

impl<F: IsField> Frame<F> {
    pub fn new(data: Vec<FieldElement<F>>, row_width: usize) -> Self {
        Self { data, row_width }
    }

    pub fn num_rows(&self) -> usize {
        self.data.len() / self.row_width
    }

    pub fn num_columns(&self) -> usize {
        self.row_width
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.row_width;
        &self.data[row_offset..row_offset + self.row_width]
    }

    pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<F>] {
        let row_offset = row_idx * self.row_width;
        &mut self.data[row_offset..row_offset + self.row_width]
    }

    pub fn offsets() -> &'static [usize] {
        &[0, 1, 2]
    }

    pub fn read_from_trace(trace: &TraceTable<F>, step: usize, blowup: u8) -> Self {
        let mut data = Vec::new();

        // Get trace length to apply module with it when getting elements of
        // the frame from the trace.
        let trace_len = trace.table.len();
        for frame_row_idx in Self::offsets().iter() {
            data.push(trace.table[step + (frame_row_idx * blowup as usize) % trace_len])
        }

        Self::new(data, 1)
    }
}
