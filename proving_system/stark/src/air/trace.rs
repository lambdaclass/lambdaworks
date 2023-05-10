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
pub struct TraceTable<F: IsFFTField> {
    /// `table` is row-major trace element description
    pub table: Vec<FieldElement<F>>,
    pub n_cols: usize,
}

impl<F: IsFFTField> TraceTable<F> {
    pub fn empty() -> Self {
        Self { table: Vec::new() , n_cols: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.n_cols == 0
    }

    pub fn new_from_cols(cols: &[Vec<FieldElement<F>>]) -> Self {
        let n_rows = cols[0].len();
        debug_assert!(cols.iter().all(|c| c.len() == n_rows));

        let n_cols = cols.len();

        let mut table = Vec::with_capacity(n_cols * n_rows);

        for row_idx in 0..n_rows {
            for col in cols {
                table.push(col[row_idx].clone());
            }
        }
        Self { table, n_cols }
    }

    pub fn n_rows(&self) -> usize {
        if self.n_cols == 0 {
            0
        } else {
            self.table.len() / self.n_cols
        }
    }

    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        let n_rows = self.n_rows();
        (0..n_rows)
            .map(|row_idx| {
                self.table[(row_idx * self.n_cols)..(row_idx * self.n_cols + self.n_cols)].to_vec()
            })
            .collect()
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.n_cols;
        &self.table[row_offset..row_offset + self.n_cols]
    }

    pub fn cols(&self) -> Vec<Vec<FieldElement<F>>> {
        let n_rows = self.n_rows();
        (0..self.n_cols)
            .map(|col_idx| {
                (0..n_rows)
                    .map(|row_idx| self.table[row_idx * self.n_cols + col_idx].clone())
                    .collect()
            })
            .collect()
    }

    /// Given a step and a column index, gives stored value in that position
    pub fn get(&self, step: usize, col: usize) -> FieldElement<F> {
        let idx = step * self.n_cols + col;
        self.table[idx].clone()
    }

    pub fn compute_trace_polys(&self) -> Vec<Polynomial<FieldElement<F>>> {
        self.cols()
            .iter()
            .map(|col| Polynomial::interpolate_fft(col))
            .collect::<Result<Vec<Polynomial<FieldElement<F>>>, FFTError>>()
            .unwrap()
    }

    /// Validates that the trace is valid with respect to the supplied AIR constraints
    pub fn validate<A: AIR<Field = F>>(&self, air: &A, rap_challenges: &A::RAPChallenges) -> bool {
        info!("Starting constraints validation over trace...");
        let mut ret = true;

        // --------- VALIDATE BOUNDARY CONSTRAINTS ------------
        air.boundary_constraints(rap_challenges)
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

            let evaluations = air.compute_transition(&frame, rap_challenges);
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

        let trace_table = TraceTable::new_from_cols(&[col_1.clone(), col_2.clone()]);
        let res_cols = trace_table.cols();

        assert_eq!(res_cols, vec![col_1, col_2]);
    }
}
