use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

use super::trace::TraceTable;

#[derive(Clone, Debug)]
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

    pub fn read_from_trace(
        trace: &TraceTable<F>,
        step: usize,
        blowup: u8,
        offsets: &[usize],
    ) -> Self {
        let mut data = Vec::new();

        // Get trace length to apply module with it when getting elements of
        // the frame from the trace.
        let trace_len = trace.table.len();
        for frame_row_idx in offsets.iter() {
            data.push(
                trace.table
                    [(step + (frame_row_idx * usize::try_from(blowup).unwrap())) % trace_len]
                    .clone(),
            )
        }

        Self::new(data, 1)
    }

    /// Returns the Out of Domain Frame for the given trace polynomials, out of domain evaluation point (called `z` in the literature),
    /// frame offsets given by the AIR and primitive root used for interpolating the trace polynomials.
    /// An out of domain frame is nothing more than the evaluation of the trace polynomials in the points required by the
    /// verifier to check the consistency between the trace and the composition polynomial.
    ///
    /// In the fibonacci example, the ood frame is simply the evaluations `[t(z), t(z * g), t(z * g^2)]`, where `t` is the trace
    /// polynomial and `g` is the primitive root of unity used when interpolating `t`.
    pub fn construct_ood_frame(
        trace_polys: &[Polynomial<FieldElement<F>>],
        z: &FieldElement<F>,
        frame_offsets: &[usize],
        primitive_root: &FieldElement<F>,
    ) -> Self {
        let mut data = vec![];
        let evaluation_points: Vec<FieldElement<F>> = frame_offsets
            .iter()
            .map(|offset| z * primitive_root.pow(*offset))
            .collect();

        for poly in trace_polys {
            data.push(poly.evaluate_slice(&evaluation_points));
        }

        Self {
            data: data.into_iter().flatten().collect(),
            row_width: trace_polys.len(),
        }
    }
}
