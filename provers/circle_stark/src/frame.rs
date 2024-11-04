use crate::trace::LDETraceTable;
use itertools::Itertools;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

/// A frame represents a collection of trace steps.
/// The collected steps are all the necessary steps for
/// all transition costraints over a trace to be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame<F: IsField>
where
    F: IsField,
{
    steps: Vec<Vec<FieldElement<F>>>,
}

impl<F: IsField> Frame<F> {

    pub fn new(steps: Vec<Vec<F>>) -> Self {
        Self { steps }
    }

    pub fn get_evaluation_step(&self, step: usize) -> &Vec<FieldElement<F>> {
        &self.steps[step]
    }

    pub fn read_from_lde(
        lde_trace: &LDETraceTable<F>,
        row: usize,
        offsets: &[usize],
    ) -> Self {
        let num_rows = lde_trace.num_rows();

        let lde_steps = offsets
            .iter()
            .map(|offset| {
                let row = lde_trace.get_row(row + offset);
            })
            .collect_vec();

        Frame::new(lde_steps)
    }
}
