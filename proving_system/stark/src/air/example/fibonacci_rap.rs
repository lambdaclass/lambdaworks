use std::ops::Div;

use crate::{
    air::{
        self,
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        context::AirContext,
        frame::Frame,
        trace::{AuxiliarySegment, TraceTable},
        AIR,
    },
    fri::FieldElement,
};
use lambdaworks_math::field::{
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::IsField,
};

#[derive(Clone)]
pub struct FibonacciRAP {
    context: AirContext,
}

impl FibonacciRAP {
    pub fn new(context: AirContext) -> Self {
        Self { context }
    }
}

impl AIR for FibonacciRAP {
    type Field = Stark252PrimeField;

    fn compute_transition(&self, frame: &Frame<Self::Field>) -> Vec<FieldElement<Self::Field>> {
        let first_row = frame.get_row(0);
        let second_row = frame.get_row(1);
        let third_row = frame.get_row(2);

        vec![third_row[0].clone() - second_row[0].clone() - first_row[0].clone()]
    }

    fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
        let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn compute_aux_transition(
        &self,
        main_frame: &Frame<Self::Field>,
        aux_frame: &Frame<Self::Field>,
        aux_rand_elements: &[FieldElement<Self::Field>],
    ) -> Vec<FieldElement<Self::Field>> {
        let z_i = aux_frame.get_row(0)[0];
        let z_i_plus_one = aux_frame.get_row(1)[0];

        let a_i = main_frame.get_row(0)[0];
        let b_i = main_frame.get_row(0)[1];

        let gamma = aux_rand_elements[0];

        let numerator = z_i * (a_i + gamma);
        let denominator = b_i + gamma;

        let eval = z_i_plus_one - numerator.div(denominator);

        vec![eval]
    }

    fn build_aux_segment(
        &self,
        trace: &TraceTable<Self::Field>,
        segment_idx: usize,
        rand_elements: &[FieldElement<Self::Field>],
    ) -> Option<AuxiliarySegment<Self::Field>> {
        let main_segment_cols = trace.main_cols();

        let not_perm = main_segment_cols[0].clone();
        let perm = main_segment_cols[1].clone();
        let gamma = rand_elements[0];

        let trace_len = trace.n_rows();

        let mut aux_col = Vec::new();
        for i in 0..trace_len {
            if i == 0 {
                aux_col.push(FieldElement::<Self::Field>::one());
            } else {
                let z_i = aux_col[i - 1];
                let n_p_term = not_perm[i - 1].clone() + gamma;
                let p_term = perm[i - 1] + gamma;

                aux_col.push(z_i * n_p_term.div(p_term));
            }
        }
        let aux_segment = AuxiliarySegment::new_from_cols(&[aux_col]);

        Some(aux_segment)
    }

    // fn aux_boundary_constraints(
    //     &self,
    //     segment_idx: usize,
    //     aux_rand_elements: &[FieldElement<Self::Field>],
    // ) -> BoundaryConstraints<Self::Field> {
    //     let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
    // }

    fn context(&self) -> AirContext {
        self.context.clone()
    }
}

pub fn fibonacci_rap_trace<F: IsField>(
    initial_values: [FieldElement<F>; 2],
    trace_length: usize,
) -> Vec<Vec<FieldElement<F>>> {
    let mut fib_seq: Vec<FieldElement<F>> = vec![];

    fib_seq.push(initial_values[0].clone());
    fib_seq.push(initial_values[1].clone());

    for i in 2..(trace_length) {
        fib_seq.push(fib_seq[i - 1].clone() + fib_seq[i - 2].clone());
    }

    let last_value = fib_seq[trace_length - 1].clone();
    let mut fib_permuted = fib_seq.clone();
    fib_permuted[0] = last_value;
    fib_permuted[trace_length - 1] = initial_values[0].clone();

    fib_seq.push(FieldElement::<F>::zero());
    fib_permuted.push(FieldElement::<F>::zero());

    vec![fib_seq, fib_permuted]
}

#[cfg(test)]
mod test {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::FE17;

    #[test]
    fn fib_rap_trace() {
        let trace = fibonacci_rap_trace([FE17::from(1), FE17::from(1)], 8);

        println!("TRACE: {:?}", trace);
    }

    #[test]
    fn aux_col() {
        let trace = fibonacci_rap_trace([FE17::from(1), FE17::from(1)], 64);

        let not_perm = trace[0].clone();
        let perm = trace[1].clone();
        let gamma = FE17::from(10);

        assert_eq!(perm.len(), not_perm.len());
        let trace_len = not_perm.len();

        let mut aux_col = Vec::new();
        for i in 0..trace_len {
            println!("IDX: {i}");
            if i == 0 {
                aux_col.push(FE17::one());
            } else {
                let z_i = aux_col[i - 1];
                let n_p_term = not_perm[i - 1] + gamma;
                let p_term = perm[i - 1] + gamma;

                aux_col.push(z_i * n_p_term.div(p_term));
            }
        }

        assert_eq!(aux_col.last().unwrap(), &FE17::one());
    }
}
