use super::trace::TraceTable;
use crate::trace::StepView;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

#[derive(Clone, Debug, PartialEq)]
// #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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

    /// Given a slice of trace polynomials, an evaluation point `x`, the frame offsets
    /// corresponding to the computation of the transitions, and a primitive root,
    /// outputs the trace evaluations of each trace polynomial over the values used to
    /// compute a transition.
    /// Example: For a simple Fibonacci computation, if t(x) is the trace polynomial of
    /// the computation, this will output evaluations t(x), t(g * x), t(g^2 * z).
    pub fn get_trace_evaluations(
        trace_polys: &[Polynomial<FieldElement<F>>],
        x: &FieldElement<F>,
        frame_offsets: &[usize],
        primitive_root: &FieldElement<F>,
    ) -> Vec<Vec<FieldElement<F>>> {
        frame_offsets
            .iter()
            .map(|offset| x * primitive_root.pow(*offset))
            .map(|eval_point| {
                trace_polys
                    .iter()
                    .map(|poly| poly.evaluate(&eval_point))
                    .collect::<Vec<FieldElement<F>>>()
            })
            .collect()
    }
}
