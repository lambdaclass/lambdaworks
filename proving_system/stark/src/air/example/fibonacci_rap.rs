use std::ops::Div;

use crate::{
    air::{
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        context::AirContext,
        frame::Frame,
        trace::TraceTable,
        AIR,
    },
    fri::FieldElement,
    transcript_to_field,
};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::{
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::IsFFTField,
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
    type RawTrace = Vec<Vec<FieldElement<Self::Field>>>;
    type RAPChallenges = FieldElement<Self::Field>;
    type PublicInput = ();

    fn build_main_trace(
        &self,
        raw_trace: &Self::RawTrace,
        _public_input: &mut Self::PublicInput,
    ) -> TraceTable<Self::Field> {
        TraceTable::new_from_cols(raw_trace)
    }

    fn build_auxiliary_trace(
        &self,
        main_trace: &TraceTable<Self::Field>,
        gamma: &Self::RAPChallenges,
        _public_input: &Self::PublicInput,
    ) -> TraceTable<Self::Field> {
        let main_segment_cols = main_trace.cols();
        let not_perm = &main_segment_cols[0];
        let perm = &main_segment_cols[1];

        let trace_len = main_trace.n_rows();

        let mut aux_col = Vec::new();
        for i in 0..trace_len {
            if i == 0 {
                aux_col.push(FieldElement::<Self::Field>::one());
            } else {
                let z_i = &aux_col[i - 1];
                let n_p_term = not_perm[i - 1].clone() + gamma;
                let p_term = &perm[i - 1] + gamma;

                aux_col.push(z_i * n_p_term.div(p_term));
            }
        }
        TraceTable::new_from_cols(&[aux_col])
    }

    fn build_rap_challenges<T: Transcript>(&self, transcript: &mut T) -> Self::RAPChallenges {
        transcript_to_field(transcript)
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        1
    }

    fn compute_transition(
        &self,
        frame: &Frame<Self::Field>,
        gamma: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        // Main constraints
        let first_row = frame.get_row(0);
        let second_row = frame.get_row(1);
        let third_row = frame.get_row(2);

        let mut constraints =
            vec![third_row[0].clone() - second_row[0].clone() - first_row[0].clone()];

        // Auxiliary constraints
        let z_i = &frame.get_row(0)[2];
        let z_i_plus_one = &frame.get_row(1)[2];

        let a_i = &frame.get_row(0)[0];
        let b_i = &frame.get_row(0)[1];

        let eval = z_i_plus_one * (b_i + gamma) - z_i * (a_i + gamma);

        constraints.push(eval);
        constraints
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &Self::RAPChallenges,
        _public_input: &Self::PublicInput,
    ) -> BoundaryConstraints<Self::Field> {
        // Main boundary constraints
        let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
        let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());

        // Auxiliary boundary constraints
        let a0_aux = BoundaryConstraint::new(2, 0, FieldElement::<Self::Field>::one());

        BoundaryConstraints::from_constraints(vec![a0, a1, a0_aux])
    }

    fn context(&self) -> AirContext {
        self.context.clone()
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.context().trace_length
    }
}

pub fn fibonacci_rap_trace<F: IsFFTField>(
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
    fn test_build_fibonacci_rap_trace() {
        // The fibonacci RAP trace should have two columns:
        //     * The usual fibonacci sequence column
        //     * The permuted fibonacci sequence column. The first and last elements are permuted.
        // Also, a 0 is appended at the end of both columns. The reason for this can be read in
        // https://hackmd.io/@aztec-network/plonk-arithmetiization-air#RAPs---PAIRs-with-interjected-verifier-randomness

        let trace = fibonacci_rap_trace([FE17::from(1), FE17::from(1)], 8);
        let expected_trace = vec![
            vec![
                FE17::one(),
                FE17::one(),
                FE17::from(2),
                FE17::from(3),
                FE17::from(5),
                FE17::from(8),
                FE17::from(13),
                FE17::from(21),
                FE17::zero(),
            ],
            vec![
                FE17::from(21),
                FE17::one(),
                FE17::from(2),
                FE17::from(3),
                FE17::from(5),
                FE17::from(8),
                FE17::from(13),
                FE17::one(),
                FE17::zero(),
            ],
        ];

        assert_eq!(trace, expected_trace);
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
