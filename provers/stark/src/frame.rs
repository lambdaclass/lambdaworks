use super::trace::TraceTable;
use crate::trace::StepView;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Frame<'t, F: IsFFTField> {
    steps: Vec<StepView<'t, F>>,
}

impl<'t, F: IsFFTField> Frame<'t, F> {
    pub fn new(steps: Vec<StepView<'t, F>>) -> Self {
        Self { steps }
    }

    pub fn get_evaluation_step(&self, step: usize) -> StepView<'t, F> {
        self.steps[step]
    }

    // pub fn n_rows(&self) -> usize {
    //     self.table.height
    // }

    // pub fn n_cols(&self) -> usize {
    //     self.table.width
    // }

    // pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
    //     self.table.get_row(row_idx)
    // }

    // pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<F>] {
    //     self.table.get_row_mut(row_idx)
    // }

    pub fn read_from_trace(
        trace: &TraceTable<F>,
        step: usize,
        blowup: u8,
        offsets: &[usize],
    ) -> Self {
        // Get trace length to apply module with it when getting elements of
        // the frame from the trace.
        let trace_steps = trace.num_steps();

        let steps = offsets
            .iter()
            .map(|eval_offset| {
                trace.step_view((step + (eval_offset * blowup as usize)) % trace_steps)
            })
            .collect();

        Self::new(steps)
    }
}

impl<F: IsFFTField> From<&Table<F>> for Frame<F> {
    fn from(value: &Table<F>) -> Self {
        Self::new(value.data.clone(), value.width)
    }
}
