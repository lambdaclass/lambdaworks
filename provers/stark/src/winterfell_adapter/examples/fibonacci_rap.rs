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
pub struct FibonacciRAP {
    context: AirContext<FieldElement<Stark252PrimeField>>,
    result: FieldElement<Stark252PrimeField>,
}

impl Air for FibonacciRAP {
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
        FibonacciRAP {
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

        // constraints of Fibonacci sequence (1 aux variable):
        result[0] = next[0] - current[1];
        result[1] = next[1] - (current[1] + current[0]);
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

    let mut fibonacci = vec![FieldElement::one(), FieldElement::one()];
    for i in 2..(sequence_length + 1) {
        fibonacci.push(fibonacci[i - 2] + fibonacci[i - 1])
    }
    
    TraceTable::init(vec![
        fibonacci[..fibonacci.len() - 1].to_vec(),
        fibonacci[1..].to_vec(),
    ])
}
