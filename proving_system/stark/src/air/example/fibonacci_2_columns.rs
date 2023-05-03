use crate::{
    air::{
        self,
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        context::AirContext,
        trace::TraceTable,
        AIR,
    },
    fri::FieldElement,
};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::{
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::IsField,
};

#[derive(Clone, Debug)]
pub struct Fibonacci2ColsAIR {
    context: AirContext,
}

impl From<AirContext> for Fibonacci2ColsAIR {
    fn from(context: AirContext) -> Self {
        Self { context }
    }
}

impl AIR for Fibonacci2ColsAIR {
    type RawTrace = Vec<Vec<FieldElement<Stark252PrimeField>>>;
    type RAPChallenges = ();

    fn build_main_trace(raw_trace: &Self::RawTrace) -> TraceTable {
        TraceTable::new_from_cols(raw_trace)
    }

    fn build_auxiliary_trace(
        _main_trace: &TraceTable,
        _rap_challenges: &Self::RAPChallenges,
    ) -> TraceTable {
        TraceTable::empty()
    }

    fn build_rap_challenges<T: Transcript>(_transcript: &mut T) -> Self::RAPChallenges {}

    fn compute_transition(
        &self,
        frame: &air::frame::Frame<Stark252PrimeField>,
        _rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Stark252PrimeField>> {
        let first_row = frame.get_row(0);
        let second_row = frame.get_row(1);

        // constraints of Fibonacci sequence (2 terms per step):
        // s_{0, i+1} = s_{0, i} + s_{1, i}
        // s_{1, i+1} = s_{1, i} + s_{0, i+1}
        let first_transition = &second_row[0] - &first_row[0] - &first_row[1];
        let second_transition = &second_row[1] - &first_row[1] - &second_row[0];

        vec![first_transition, second_transition]
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &Self::RAPChallenges,
    ) -> BoundaryConstraints<Stark252PrimeField> {
        let a0 = BoundaryConstraint::new(0, 0, FieldElement::one());
        let a1 = BoundaryConstraint::new(1, 0, FieldElement::one());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn context(&self) -> air::context::AirContext {
        self.context.clone()
    }
}

pub fn fibonacci_trace_2_columns<F: IsField>(
    initial_values: [FieldElement<F>; 2],
    trace_length: usize,
) -> Vec<Vec<FieldElement<F>>> {
    let mut ret1: Vec<FieldElement<F>> = vec![];
    let mut ret2: Vec<FieldElement<F>> = vec![];

    ret1.push(initial_values[0].clone());
    ret2.push(initial_values[1].clone());

    for i in 1..(trace_length) {
        let new_val = ret1[i - 1].clone() + ret2[i - 1].clone();
        ret1.push(new_val.clone());
        ret2.push(new_val + ret2[i - 1].clone());
    }

    vec![ret1, ret2]
}
