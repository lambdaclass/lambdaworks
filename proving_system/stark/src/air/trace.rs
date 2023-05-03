use crate::air::frame::Frame;
use crate::air::AIR;
use lambdaworks_fft::errors::FFTError;
use lambdaworks_fft::polynomial::FFTPoly;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
use log::{error, info};

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct AuxiliarySegment<F: IsFFTField> {
    //`aux_segment` is a row-major trace element description
    pub aux_segment: Vec<FieldElement<F>>,
    pub aux_segment_width: usize,
}

impl<F: IsFFTField> AuxiliarySegment<F> {
    fn n_rows(&self) -> usize {
        self.aux_segment.len() / self.aux_segment_width
    }

    fn cols(&self) -> Vec<Vec<FieldElement<F>>> {
        let n_rows = self.n_rows();
        (0..self.aux_segment_width)
            .map(|col_idx| {
                (0..n_rows)
                    .map(|row_idx| {
                        self.aux_segment[row_idx * self.aux_segment_width + col_idx].clone()
                    })
                    .collect()
            })
            .collect()
    }

    pub fn compute_aux_segment_polys(&self) -> Vec<Polynomial<FieldElement<F>>> {
        self.cols()
            .iter()
            .map(|col| Polynomial::interpolate_fft(col))
            .collect::<Result<Vec<Polynomial<FieldElement<F>>>, FFTError>>()
            .unwrap()
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct TraceTable<F: IsFFTField> {
    // `main_segment` is row-major trace element description
    pub main_segment: Vec<FieldElement<F>>,
    pub main_segment_width: usize,
    pub aux_segments: Option<Vec<AuxiliarySegment<F>>>,
}

impl<F: IsFFTField> TraceTable<F> {
    pub fn new_from_cols(
        cols: &[Vec<FieldElement<F>>],
        aux_segments: Option<Vec<AuxiliarySegment<F>>>,
    ) -> Self {
        let n_rows = cols[0].len();
        debug_assert!(cols.iter().all(|c| c.len() == n_rows));

        let main_segment_width = cols.len();

        let mut main_segment = Vec::with_capacity(main_segment_width * n_rows);

        for row_idx in 0..n_rows {
            for col in cols {
                main_segment.push(col[row_idx].clone());
            }
        }
        Self {
            main_segment,
            main_segment_width,
            aux_segments,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.main_segment.len() / self.main_segment_width
    }

    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        let n_rows = self.n_rows();
        (0..n_rows)
            .map(|row_idx| {
                self.main_segment[(row_idx * self.main_segment_width)
                    ..(row_idx * self.main_segment_width + self.main_segment_width)]
                    .to_vec()
            })
            .collect()
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.main_segment_width;
        &self.main_segment[row_offset..row_offset + self.main_segment_width]
    }

    pub fn main_cols(&self) -> Vec<Vec<FieldElement<F>>> {
        let n_rows = self.n_rows();
        (0..self.main_segment_width)
            .map(|col_idx| {
                (0..n_rows)
                    .map(|row_idx| {
                        self.main_segment[row_idx * self.main_segment_width + col_idx].clone()
                    })
                    .collect()
            })
            .collect()
    }

    /// Given a step and a column index, gives stored value in that position
    pub fn get(&self, step: usize, col: usize) -> FieldElement<F> {
        let idx = step * self.main_segment_width + col;
        self.main_segment[idx].clone()
    }

    pub fn compute_trace_polys(&self) -> Vec<Polynomial<FieldElement<F>>> {
        self.main_cols()
            .iter()
            .map(|col| Polynomial::interpolate_fft(col))
            .collect::<Result<Vec<Polynomial<FieldElement<F>>>, FFTError>>()
            .unwrap()
    }

    /// Validates that the trace is valid with respect to the supplied AIR constraints
    pub fn validate<A: AIR<Field = F>>(&self, air: &A) -> bool {
        info!("Starting constraints validation over trace...");
        let mut ret = true;

        // --------- VALIDATE BOUNDARY CONSTRAINTS ------------
        air.boundary_constraints()
            .constraints
            .iter()
            .for_each(|constraint| {
                let col = constraint.col;
                let step = constraint.step;
                let boundary_value = constraint.value.clone();
                let trace_value = self.get(step, col);

                if boundary_value != trace_value {
                    ret = false;
                    error!("Boundary constraint inconsistency - Expected value {:?} in step {} and column {}, found: {:?}", boundary_value, step, col, trace_value);
                }
            });

        // --------- VALIDATE TRANSITION CONSTRAINTS -----------
        let n_transition_constraints = air.context().num_transition_constraints();
        let transition_exemptions = air.context().transition_exemptions;

        let exemption_steps: Vec<usize> = vec![self.n_rows(); n_transition_constraints]
            .iter()
            .zip(transition_exemptions)
            .map(|(trace_steps, exemptions)| trace_steps - exemptions)
            .collect();

        // Iterate over trace and compute transitions
        for step in 0..self.n_rows() {
            let frame = Frame::read_from_trace(self, step, 1, &air.context().transition_offsets);

            let evaluations = air.compute_transition(&frame);
            // Iterate over each transition evaluation. When the evaluated step is not from
            // the exemption steps corresponding to the transition, it should have zero as a
            // result
            evaluations.iter().enumerate().for_each(|(i, eval)| {
                if step < exemption_steps[i] && eval != &FieldElement::<F>::zero() {
                    ret = false;
                    error!(
                        "Inconsistent evaluation of transition {} in step {} - expected 0, got {:?}", i, step, eval
                    );
                }
            })
        }
        info!("Constraints validation check ended");
        ret
    }
}

#[cfg(test)]
mod test {
    use super::TraceTable;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::F17};

    #[test]
    fn test_cols() {
        type F = FieldElement<F17>;

        let col_1 = vec![F::from(1), F::from(2), F::from(5), F::from(13)];
        let col_2 = vec![F::from(1), F::from(3), F::from(8), F::from(21)];

        let trace_table = TraceTable::new_from_cols(&[col_1.clone(), col_2.clone()], None);
        let res_cols = trace_table.main_cols();

        assert_eq!(res_cols, vec![col_1, col_2]);
    }
}
