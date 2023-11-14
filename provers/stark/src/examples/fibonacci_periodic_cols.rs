use std::fmt::Display;

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

/// A fibonacci sequence with a twist. It has two columns
/// - C1: at each step adds the last two values or does
///       nothing depending on C2.
/// - C2: it is a binary column that cycles around [0, 1]
/// 
///   C1   |   C2
///   1    |   0     Boundary col1 = 1
///   1    |   1     Boundary col1 = 1
///   1    |   0     Does nothing
///   2    |   1     Adds 1 + 1
///   2    |   0     Does nothing
///   4    |   1     Adds 2 + 2
///   4    |   0     ...
///   8    |   1
#[derive(Clone)]
pub struct FibonacciPeriodicAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    pub_inputs: FibonacciPeriodicPublicInputs<F>,
}

#[derive(Clone, Debug)]
pub struct FibonacciPeriodicPublicInputs<F>
where
    F: IsFFTField,
{
    pub a0: FieldElement<F>,
    pub a1: FieldElement<F>,
}

impl<F> AIR for FibonacciPeriodicAIR<F>
where
    F: IsFFTField,
{
    type Field = F;
    type RAPChallenges = ();
    type PublicInputs = FibonacciPeriodicPublicInputs<Self::Field>;

    const STEP_SIZE: usize = 1;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 1,
            transition_degrees: vec![1],
            transition_exemptions: vec![2],
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: 1,
            num_transition_exemptions: 1,
        };

        Self {
            pub_inputs: pub_inputs.clone(),
            context,
            trace_length,
        }
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length()
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
        periodic_values: &[FieldElement<Self::Field>],
        _rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);
        let third_step = frame.get_evaluation_step(2);

        let a0 = first_step.get_evaluation_element(0, 0);
        let a1 = second_step.get_evaluation_element(0, 0);
        let a2 = third_step.get_evaluation_element(0, 0);

        let s = &periodic_values[0];

        vec![s * (a2 - a1 - a0)]
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &Self::RAPChallenges,
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple(0, self.pub_inputs.a0.clone());
        let a1 = BoundaryConstraint::new_simple(self.trace_length() - 1, self.pub_inputs.a1.clone());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        0
    }

    fn get_periodic_column_values(&self) -> Vec<Vec<FieldElement<Self::Field>>> {
        vec![vec![FieldElement::zero(), FieldElement::one()]]
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}

pub fn fibonacci_trace<F: IsFFTField>(
    trace_length: usize,
) -> TraceTable<F>
{
    let mut ret: Vec<FieldElement<F>> = vec![];

    ret.push(FieldElement::one());
    ret.push(FieldElement::one());
    ret.push(FieldElement::one());

    let mut accum = FieldElement::from(2);
    while ret.len() < trace_length - 1 {
        ret.push(accum.clone());
        ret.push(accum.clone());
        accum = &accum + &accum;
    }
    ret.push(accum);

    TraceTable::from_columns(vec![ret], 1)
}
