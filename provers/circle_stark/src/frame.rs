use crate::trace::LDETraceTable;
use itertools::Itertools;
use lambdaworks_math::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};

/// A frame represents a collection of trace steps.
/// The collected steps are all the necessary steps for
/// all transition costraints over a trace to be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    steps: Vec<Vec<FieldElement<Mersenne31Field>>>,
}

impl Frame {
    pub fn new(steps: Vec<Vec<FieldElement<Mersenne31Field>>>) -> Self {
        Self { steps }
    }

    pub fn get_evaluation_step(&self, step: usize) -> &Vec<FieldElement<Mersenne31Field>> {
        &self.steps[step]
    }

    pub fn read_from_lde(lde_trace: &LDETraceTable, row: usize, offsets: &[usize]) -> Self {
        let num_rows = lde_trace.num_rows();

        let lde_steps = offsets
            .iter()
            .map(|offset| lde_trace.get_row((row + offset) % num_rows).to_vec())
            .collect_vec();

        Frame::new(lde_steps)
    }
}
