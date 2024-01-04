use crate::{
    table::TableView,
    trace::{LDETraceTable, StepView, TraceTable},
};
use itertools::Itertools;
use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};

/// A frame represents a collection of trace steps.
/// The collected steps are all the necessary steps for
/// all transition costraints over a trace to be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame<'t, F: IsFFTField> {
    steps: Vec<TableView<'t, F>>,
}

impl<'t, F: IsFFTField> Frame<'t, F> {
    pub fn new(steps: Vec<TableView<'t, F>>) -> Self {
        Self { steps }
    }

    pub fn get_evaluation_step(&self, step: usize) -> &TableView<'t, F> {
        &self.steps[step]
    }

    pub fn read_from_trace(trace: &'t TraceTable<F>, step: usize, offsets: &[usize]) -> Self {
        // Get trace length to apply module with it when getting elements of
        // the frame from the trace.
        let trace_rows = trace.n_rows();
        let step_size = trace.step_size;

        // let steps = offsets
        //     .iter()
        //     .map(|eval_offset| {
        //         // trace.step_view((step + (eval_offset * blowup as usize)) % trace_steps)
        //         trace.step_view((step + eval_offset) % trace_steps)
        //     })
        //     .collect();

        let row = trace.step_to_row(step);

        let steps = offsets
            .iter()
            .map(|offset| {
                let initial_step_row = row + offset * step_size;
                let end_step_row = initial_step_row + step_size;
                let table_view_data = (initial_step_row..end_step_row)
                    .map(|step_row| {
                        let step_row = step_row % trace_rows;
                        trace.get_row(step_row)
                    })
                    .collect_vec();

                TableView::new(table_view_data, trace.n_cols(), trace_rows)
            })
            .collect_vec();

        Self::new(steps)
    }

    pub fn read_from_lde(lde_trace: &'t LDETraceTable<F>, row: usize, offsets: &[usize]) -> Self {
        let blowup_factor = lde_trace.blowup_factor;
        let num_rows = lde_trace.num_rows();
        let num_cols = lde_trace.num_cols();
        let step_size = lde_trace.lde_step_size;

        // println!("READ FROM LDE - ROW: {}", row);

        let lde_steps = offsets
            .iter()
            .map(|offset| {
                // println!();
                let initial_step_row = row + offset * step_size;
                let end_step_row = initial_step_row + step_size;
                // println!("INITIAL STEP ROW: {}", initial_step_row);
                // println!("END STEP ROW: {}", end_step_row);
                let data = (initial_step_row..end_step_row)
                    .step_by(blowup_factor)
                    .map(|step_row| {
                        let step_row = step_row % num_rows;
                        // println!("STEP ROW: {}", step_row);
                        lde_trace.get_row(step_row)
                    })
                    .collect_vec();
                TableView::new(data, num_cols, num_rows)
            })
            .collect_vec();

        // println!();

        Frame::new(lde_steps)
    }
}
