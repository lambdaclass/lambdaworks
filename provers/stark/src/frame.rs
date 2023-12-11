use super::trace::TraceTable;
use crate::trace::StepView;
use lambdaworks_math::field::traits::{IsFFTField, IsField};

/// A frame represents a collection of trace steps.
/// The collected steps are all the necessary steps for
/// all transition costraints over a trace to be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame<'t, F: IsField> {
    steps: Vec<StepView<'t, F>>,
}

impl<'t, F: IsField> Frame<'t, F> {
    pub fn new(steps: Vec<StepView<'t, F>>) -> Self {
        Self { steps }
    }

    pub fn get_evaluation_step(&self, step: usize) -> &StepView<F> {
        &self.steps[step]
    }

    pub fn read_from_trace(
        trace: &'t TraceTable<F>,
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
