use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

use super::trace::TraceTable;

#[derive(Clone, Debug)]
pub struct Frame<F: IsField> {
    // Vector of rows
    data: Vec<FieldElement<F>>,
    row_width: usize,
}

impl<F: IsField> Frame<F> {
    pub fn new(data: Vec<FieldElement<F>>, row_width: usize) -> Self {
        Self {
            data: data,
            row_width,
        }
    }

    // pub fn ood_new(data_input: &[Vec<FieldElement<F>>], row_width: usize) -> Self {
    //     let mut data = Vec::new();
    //     for row in data_input {
    //         let mut row_vec = Vec::new();
    //         for elem in row {
    //             row_vec.push(elem);
    //         }
    //         data.push(row_vec);
    //     }

    //     Self { data, row_width }
    // }

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
        let mut rows = Vec::with_capacity(offsets.len());

        // Get trace length to apply module with it when getting elements of
        // the frame from the trace.
        let trace_steps = trace.n_rows();
        for frame_row_idx in offsets.iter() {
            let row = trace.get_row((step + (frame_row_idx * blowup as usize)) % trace_steps);
            rows.push(row.to_vec());
        }
        // TODO: Create `data` inside the for loop to avoid cloning again
        let data = rows.into_iter().flatten().collect();
        Self::new(data, trace.n_cols)
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
        // let mut evaluations = Vec::with_capacity(frame_offsets.len());
        // let evaluations: Vec<Vec<FieldElement<F>>> = frame_offsets
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

        // trace_polys
        //     .iter()
        //     .for_each(|p| evaluations.push(p.evaluate_slice(&evaluation_points)));

        // evaluations
    }

    // / Returns the Out of Domain Frame for the given trace polynomials, out of domain evaluation point (called `z` in the literature),
    // / frame offsets given by the AIR and primitive root used for interpolating the trace polynomials.
    // / An out of domain frame is nothing more than the evaluation of the trace polynomials in the points required by the
    // / verifier to check the consistency between the trace and the composition polynomial.
    // /
    // / In the fibonacci example, the ood frame is simply the evaluations `[t(z), t(z * g), t(z * g^2)]`, where `t` is the trace
    // / polynomial and `g` is the primitive root of unity used when interpolating `t`.
    // pub fn construct_ood_frame(
    //     trace_polys: &[Polynomial<FieldElement<F>>],
    //     z: &FieldElement<F>,
    //     frame_offsets: &[usize],
    //     primitive_root: &FieldElement<F>,
    // ) -> Self {
    //     let data = Self::get_trace_evaluations(trace_polys, z, frame_offsets, primitive_root);

    //     Self {
    //         data,
    //         row_width: trace_polys.len(),
    //     }
    // }
}
