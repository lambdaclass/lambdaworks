use lambdaworks_math::field::element::FieldElement;
use winterfell::math::FieldElement as IsWinterfellFieldElement;
use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo, TraceTable,
    TransitionConstraintDegree,
};

use crate::field_element::element::AdapterFieldElement;

/// A fibonacci winterfell AIR example. Two terms are computed
/// at each step. This was taken from the original winterfell
/// repository and adapted to work with lambdaworks.
#[derive(Clone)]
pub struct FibAir2Terms {
    context: AirContext<AdapterFieldElement>,
    result: AdapterFieldElement,
}

impl Air for FibAir2Terms {
    type BaseField = AdapterFieldElement;
    type PublicInputs = AdapterFieldElement;

    fn new(trace_info: TraceInfo, pub_inputs: Self::BaseField, options: ProofOptions) -> Self {
        let degrees = vec![
            TransitionConstraintDegree::new(1),
            TransitionConstraintDegree::new(1),
        ];
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

        // Constraints of Fibonacci sequence (2 terms per step):
        // s_{0, i+1} = s_{0, i} + s_{1, i}
        // s_{1, i+1} = s_{1, i} + s_{0, i+1}
        result[0] = next[0] - (current[0] + current[1]);
        result[1] = next[1] - (current[1] + next[0]);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // A valid Fibonacci sequence should start with two ones and terminate with
        // the expected result
        let last_step = self.trace_length() - 1;
        vec![
            Assertion::single(0, 0, Self::BaseField::ONE),
            Assertion::single(1, 0, Self::BaseField::ONE),
            Assertion::single(1, last_step, self.result),
        ]
    }
}

pub fn build_trace(sequence_length: usize) -> TraceTable<AdapterFieldElement> {
    assert!(
        sequence_length.is_power_of_two(),
        "sequence length must be a power of 2"
    );

    let mut trace = TraceTable::new(2, sequence_length / 2);
    trace.fill(
        |state| {
            state[0] = AdapterFieldElement(FieldElement::one());
            state[1] = AdapterFieldElement(FieldElement::one());
        },
        |_, state| {
            state[0] += state[1];
            state[1] += state[0];
        },
    );

    trace
}
