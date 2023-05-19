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
use lambdaworks_math::field::fields::u64_prime_field::F17;

#[derive(Clone)]
pub struct Fibonacci17AIR {
    context: AirContext,
}

impl From<AirContext> for Fibonacci17AIR {
    fn from(context: AirContext) -> Self {
        Self { context }
    }
}

impl AIR for Fibonacci17AIR {
    type Field = F17;
    type RawTrace = Vec<Vec<FieldElement<Self::Field>>>;
    type RAPChallenges = ();

    fn build_main_trace(raw_trace: &Self::RawTrace) -> TraceTable<Self::Field> {
        TraceTable::new_from_cols(raw_trace).unwrap()
    }

    fn build_auxiliary_trace(
        _main_trace: &TraceTable<Self::Field>,
        _rap_challenges: &Self::RAPChallenges,
    ) -> TraceTable<Self::Field> {
        TraceTable::empty()
    }

    fn build_rap_challenges<T: Transcript>(_transcript: &mut T) -> Self::RAPChallenges {}

    fn compute_transition(
        &self,
        frame: &air::frame::Frame<Self::Field>,
        _rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        let first_row = frame.get_row(0).unwrap();
        let second_row = frame.get_row(1).unwrap();
        let third_row = frame.get_row(2).unwrap();

        vec![third_row[0] - second_row[0] - first_row[0]]
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &Self::RAPChallenges,
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
        let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());
        let result = BoundaryConstraint::new_simple(3, FieldElement::<Self::Field>::from(3));

        BoundaryConstraints::from_constraints(vec![a0, a1, result])
    }

    fn context(&self) -> air::context::AirContext {
        self.context.clone()
    }
}
