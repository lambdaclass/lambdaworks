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

#[derive(Clone)]
pub struct DummyAIR {
    context: AirContext,
}

impl From<AirContext> for DummyAIR {
    fn from(context: AirContext) -> Self {
        Self { context }
    }
}

impl AIR for DummyAIR {
    type Field = Stark252PrimeField;
    // type Field = F17;
    type RawTrace = Vec<Vec<FieldElement<Self::Field>>>;
    type RAPChallenges = ();
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
        _main_trace: &TraceTable<Self::Field>,
        _rap_challenges: &Self::RAPChallenges,
        _public_input: &Self::PublicInput,
    ) -> TraceTable<Self::Field> {
        TraceTable::empty()
    }

    fn build_rap_challenges<T: Transcript>(&self, _transcript: &mut T) -> Self::RAPChallenges {}
    fn compute_transition(
        &self,
        frame: &air::frame::Frame<Self::Field>,
        _rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        let first_row = frame.get_row(0);
        let second_row = frame.get_row(1);
        let third_row = frame.get_row(2);

        let f_constraint = &first_row[0] * (&first_row[0] - FieldElement::one());

        let fib_constraint = &third_row[1] - &second_row[1] - &first_row[1];

        vec![f_constraint, fib_constraint]
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &Self::RAPChallenges,
        _public_input: &Self::PublicInput,
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new(1, 0, FieldElement::<Self::Field>::one());
        let a1 = BoundaryConstraint::new(1, 1, FieldElement::<Self::Field>::one());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        0
    }

    fn context(&self) -> air::context::AirContext {
        self.context.clone()
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.context().trace_length
    }
}

pub fn dummy_trace<F: IsField>(trace_length: usize) -> Vec<Vec<FieldElement<F>>> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    let a0 = FieldElement::one();
    let a1 = FieldElement::one();

    ret.push(a0);
    ret.push(a1);

    for i in 2..(trace_length) {
        ret.push(ret[i - 1].clone() + ret[i - 2].clone());
    }

    vec![vec![FieldElement::<F>::one(); trace_length], ret]
}
