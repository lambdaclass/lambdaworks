
use crate::{
    table::{LDETable, OODTable},
    trace::StepView,
};
use lambdaworks_math::field::traits::{IsField, IsSubFieldOf};

/// A frame represents a collection of trace steps.
/// The collected steps are all the necessary steps for
/// all transition costraints over a trace to be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame<'t, F: IsSubFieldOf<E>, E: IsField> {
    steps: Vec<StepView<'t, F, E>>,
}

impl<'t, F: IsSubFieldOf<E>, E: IsField> Frame<'t, F, E> {
    pub fn new(steps: Vec<StepView<'t, F, E>>) -> Self {
        Self { steps }
    }

    pub fn get_evaluation_step(&self, step: usize) -> &StepView<F, E> {
        &self.steps[step]
    }

    pub fn read_from_lde_table(
        lde_table: &'t LDETable<F, E>,
        step: usize,
        blowup: u8,
        offsets: &[usize],
    ) -> Self {
        // Get trace length to apply module with it when getting elements of
        // the frame from the trace.
        let trace_steps = lde_table.num_steps();

        let steps = offsets
            .iter()
            .map(|eval_offset| {
                lde_table.step_view((step + (eval_offset * blowup as usize)) % trace_steps)
            })
            .collect();

        Self::new(steps)
    }
}

impl<'t, E: IsField> Frame<'t, E, E> {
    pub fn read_from_ood_table(ood_table: &'t OODTable<E>, offsets: &[usize]) -> Self {
        let steps: Vec<_> = offsets
            .iter()
            .map(|offset| ood_table.step_view(*offset))
            .collect();
        Self::new(steps)
    }
}
