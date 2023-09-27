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

#[derive(Clone, Debug)]
pub struct PublicInputs<F>
where
    F: IsFFTField,
{
    pub claimed_value: FieldElement<F>,
    pub claimed_index: usize,
}

#[derive(Clone, Debug)]
pub struct Fibonacci2ColsShifted<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    pub_inputs: PublicInputs<F>,
}

/// The AIR for to a 2 column trace, where each column is a Fibonacci sequence and the
/// second column is constrained to be the shift of the first one. That is, if `Col0_i`
/// and `Col1_i` denote the i-th entry of each column, then `Col0_{i+1}` equals `Col1_{i}`
/// for all `i`. Also, `Col0_0` is constrained to be `1`.
impl<F> AIR for Fibonacci2ColsShifted<F>
where
    F: IsFFTField,
{
    type Field = F;
    type RAPChallenges = ();
    type PublicInputs = PublicInputs<Self::Field>;

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

        let first_transition = &second_row[0] - &first_row[1];
        let second_transition = &second_row[1] - &first_row[0] - &first_row[1];

        vec![first_transition, second_transition]
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        0
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &Self::RAPChallenges,
    ) -> BoundaryConstraints<Self::Field> {
        let initial_condition = BoundaryConstraint::new(0, 0, FieldElement::one());
        let claimed_value_constraint = BoundaryConstraint::new(
            0,
            self.pub_inputs.claimed_index,
            self.pub_inputs.claimed_value.clone(),
        );

        BoundaryConstraints::from_constraints(vec![initial_condition, claimed_value_constraint])
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
    initial_value: FieldElement<F>,
    trace_length: usize,
) -> TraceTable<F> {
    let mut x = FieldElement::one();
    let mut y = initial_value;
    let mut col0 = vec![x.clone()];
    let mut col1 = vec![y.clone()];

    for _ in 1..trace_length {
        (x, y) = (y.clone(), &x + &y);
        col0.push(x.clone());
        col1.push(y.clone());
    }

    TraceTable::new_from_cols(&[col0, col1])
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    use super::compute_trace;

    #[test]
    fn trace_has_expected_rows() {
        let trace = compute_trace(FieldElement::<Stark252PrimeField>::one(), 8);
        assert_eq!(trace.n_rows(), 8);

        let trace = compute_trace(FieldElement::<Stark252PrimeField>::one(), 64);
        assert_eq!(trace.n_rows(), 64);
    }

    #[test]
    fn trace_of_8_rows_is_correctly_calculated() {
        let trace = compute_trace(FieldElement::<Stark252PrimeField>::one(), 8);
        assert_eq!(
            trace.get_row(0),
            vec![FieldElement::one(), FieldElement::one()]
        );
        assert_eq!(
            trace.get_row(1),
            vec![FieldElement::one(), FieldElement::from(2)]
        );
        assert_eq!(
            trace.get_row(2),
            vec![FieldElement::from(2), FieldElement::from(3)]
        );
        assert_eq!(
            trace.get_row(3),
            vec![FieldElement::from(3), FieldElement::from(5)]
        );
        assert_eq!(
            trace.get_row(4),
            vec![FieldElement::from(5), FieldElement::from(8)]
        );
        assert_eq!(
            trace.get_row(5),
            vec![FieldElement::from(8), FieldElement::from(13)]
        );
        assert_eq!(
            trace.get_row(6),
            vec![FieldElement::from(13), FieldElement::from(21)]
        );
        assert_eq!(
            trace.get_row(7),
            vec![FieldElement::from(21), FieldElement::from(34)]
        );
    }
}
