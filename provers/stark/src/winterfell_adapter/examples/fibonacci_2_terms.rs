use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use winterfell::math::FieldElement as IsWinterfellFieldElement;
use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo, TraceTable,
    TransitionConstraintDegree,
};

pub const TRACE_WIDTH: usize = 2;

#[derive(Clone)]
pub struct FibAir2Terms {
    context: AirContext<FieldElement<Stark252PrimeField>>,
    result: FieldElement<Stark252PrimeField>,
}

impl Air for FibAir2Terms {
    type BaseField = FieldElement<Stark252PrimeField>;
    type PublicInputs = FieldElement<Stark252PrimeField>;

    // CONSTRUCTOR
    // --------------------------------------------------------------------------------------------
    fn new(trace_info: TraceInfo, pub_inputs: Self::BaseField, options: ProofOptions) -> Self {
        let degrees = vec![
            TransitionConstraintDegree::new(1),
            TransitionConstraintDegree::new(1),
        ];
        assert_eq!(TRACE_WIDTH, trace_info.width());
        FibAir2Terms {
            context: AirContext::new(trace_info, degrees, 3, options),
            result: pub_inputs,
        }
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }

    fn evaluate_transition<E: IsWinterfellFieldElement + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();
        // expected state width is 2 field elements
        debug_assert_eq!(TRACE_WIDTH, current.len());
        debug_assert_eq!(TRACE_WIDTH, next.len());

        // constraints of Fibonacci sequence (2 terms per step):
        // s_{0, i+1} = s_{0, i} + s_{1, i}
        // s_{1, i+1} = s_{1, i} + s_{0, i+1}
        result[0] = next[0] - (current[0] + current[1]);
        result[1] = next[1] - (current[1] + next[0]);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // a valid Fibonacci sequence should start with two ones and terminate with
        // the expected result
        let last_step = self.trace_length() - 1;
        vec![
            Assertion::single(0, 0, Self::BaseField::ONE),
            Assertion::single(1, 0, Self::BaseField::ONE),
            Assertion::single(1, last_step, self.result),
        ]
    }
}

pub fn build_trace(sequence_length: usize) -> TraceTable<FieldElement<Stark252PrimeField>> {
    assert!(
        sequence_length.is_power_of_two(),
        "sequence length must be a power of 2"
    );

    let mut trace = TraceTable::new(TRACE_WIDTH, sequence_length / 2);
    trace.fill(
        |state| {
            state[0] = FieldElement::one();
            state[1] = FieldElement::one();
        },
        |_, state| {
            state[0] += state[1];
            state[1] += state[0];
        },
    );

    trace
}
