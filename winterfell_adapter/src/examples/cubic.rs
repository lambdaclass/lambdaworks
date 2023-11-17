use miden_core::Felt;
use winter_air::{
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree,
};
use winter_math::FieldElement as IsWinterfellFieldElement;
use winter_prover::TraceTable;

/// A fibonacci winterfell AIR example. Two terms are computed
/// at each step. This was taken from the original winterfell
/// repository and adapted to work with lambdaworks.
#[derive(Clone)]
pub struct Cubic {
    context: AirContext<Felt>,
    result: Felt,
}

impl Air for Cubic {
    type BaseField = Felt;
    type PublicInputs = Felt;

    fn new(trace_info: TraceInfo, pub_inputs: Self::BaseField, options: ProofOptions) -> Self {
        let degrees = vec![TransitionConstraintDegree::new(3)];
        Cubic {
            context: AirContext::new(trace_info, degrees, 2, options),
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

        // s_i = (s_{i-1})³
        result[0] = next[0] - (current[0] * current[0] * current[0]);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // A valid Fibonacci sequence should start with two ones and terminate with
        // the expected result
        let last_step = self.trace_length() - 1;
        vec![
            Assertion::single(0, 0, Self::BaseField::from(2u16)),
            Assertion::single(0, last_step, self.result),
        ]
    }
}

pub fn build_trace(sequence_length: usize) -> TraceTable<Felt> {
    assert!(
        sequence_length.is_power_of_two(),
        "sequence length must be a power of 2"
    );

    let mut accum = Felt::from(2u16);
    let mut column = vec![accum];
    while column.len() < sequence_length {
        accum = accum * accum * accum;
        column.push(accum);
    }
    TraceTable::init(vec![column])
}
