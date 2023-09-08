use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};

use crate::starks::{
    constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
    context::AirContext,
    frame::Frame,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::AIR,
};

#[derive(Clone)]
pub struct QuadraticAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    pub_inputs: QuadraticPublicInputs<F>,
}

#[derive(Clone, Debug)]
pub struct QuadraticPublicInputs<F>
where
    F: IsFFTField,
{
    pub a0: FieldElement<F>,
}

impl<F> AIR for QuadraticAIR<F>
where
    F: IsFFTField,
{
    type Field = F;
    type RAPChallenges = ();
    type PublicInputs = QuadraticPublicInputs<Self::Field>;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 1,
            transition_degrees: vec![2],
            transition_exemptions: vec![1],
            transition_offsets: vec![0, 1],
            num_transition_constraints: 1,
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

    fn build_rap_challenges<T: Transcript>(&self, _transcript: &mut T) -> Self::RAPChallenges {}

    fn compute_transition(
        &self,
        frame: &Frame<Self::Field>,
        _rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        let first_row = frame.get_row(0);
        let second_row = frame.get_row(1);

        vec![&second_row[0] - &first_row[0] * &first_row[0]]
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        0
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &Self::RAPChallenges,
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple(0, self.pub_inputs.a0.clone());

        BoundaryConstraints::from_constraints(vec![a0])
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self) -> usize {
        2 * self.trace_length()
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}

pub fn quadratic_trace<F: IsFFTField>(
    initial_value: FieldElement<F>,
    trace_length: usize,
) -> TraceTable<F> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    ret.push(initial_value);

    for i in 1..(trace_length) {
        ret.push(ret[i - 1].clone() * ret[i - 1].clone());
    }

    TraceTable::new_from_cols(&[ret])
}
