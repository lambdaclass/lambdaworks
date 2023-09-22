use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};

use crate::{
    constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
    context::AirContext,
    frame::Frame,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::AIR,
    transcript::IsStarkTranscript,
};

use super::simple_fibonacci::FibonacciPublicInputs;

#[derive(Clone, Debug)]
pub struct Fibonacci2ColsAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    pub_inputs: FibonacciPublicInputs<F>,
}

/// The AIR for to a 2 column trace, where the columns form a Fibonacci sequence when
/// stacked in row-major order.
impl<F> AIR for Fibonacci2ColsAIR<F>
where
    F: IsFFTField,
{
    type Field = F;
    type RAPChallenges = ();
    type PublicInputs = FibonacciPublicInputs<Self::Field>;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let context = AirContext {
            proof_options: proof_options.clone(),
            transition_degrees: vec![1, 1],
            transition_exemptions: vec![1, 1],
            transition_offsets: vec![0, 1],
            num_transition_constraints: 2,
            trace_columns: 2,
            num_transition_exemptions: 1,
        };

        Self {
            trace_length,
            context,
            pub_inputs: pub_inputs.clone(),
        }
    }

    fn build_auxiliary_trace(
        &self,
        _main_trace: &TraceTable<Self::Field>,
        _rap_challenges: &Self::RAPChallenges,
    ) -> TraceTable<Self::Field> {
        TraceTable::empty()
    }

    fn build_rap_challenges(
        &self,
        _transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> Self::RAPChallenges {
    }

    fn compute_transition(
        &self,
        frame: &Frame<Self::Field>,
        _rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        let first_row = frame.get_row(0);
        let second_row = frame.get_row(1);

        // constraints of Fibonacci sequence (2 terms per step):
        // s_{0, i+1} = s_{0, i} + s_{1, i}
        // s_{1, i+1} = s_{1, i} + s_{0, i+1}
        let first_transition = &second_row[0] - &first_row[0] - &first_row[1];
        let second_transition = &second_row[1] - &first_row[1] - &second_row[0];

        vec![first_transition, second_transition]
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        0
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &Self::RAPChallenges,
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new(0, 0, self.pub_inputs.a0.clone());
        let a1 = BoundaryConstraint::new(1, 0, self.pub_inputs.a1.clone());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length()
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}

pub fn compute_trace<F: IsFFTField>(
    initial_values: [FieldElement<F>; 2],
    trace_length: usize,
) -> TraceTable<F> {
    let mut ret1: Vec<FieldElement<F>> = vec![];
    let mut ret2: Vec<FieldElement<F>> = vec![];

    ret1.push(initial_values[0].clone());
    ret2.push(initial_values[1].clone());

    for i in 1..(trace_length) {
        let new_val = ret1[i - 1].clone() + ret2[i - 1].clone();
        ret1.push(new_val.clone());
        ret2.push(new_val + ret2[i - 1].clone());
    }

    TraceTable::new_from_cols(&[ret1, ret2])
}
