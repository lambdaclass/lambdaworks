use crate::{
    air::{
        self,
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        context::AirContext,
        AIR,
    },
    fri::FieldElement,
};
use lambdaworks_math::field::{
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::IsField,
};

#[derive(Clone)]
pub struct FibonacciAIR {
    context: AirContext,
}

impl AIR for FibonacciAIR {
    type Field = Stark252PrimeField;

    fn new(context: air::context::AirContext) -> Self {
        Self { context }
    }

    fn compute_transition(
        &self,
        frame: &air::frame::Frame<Self::Field>,
    ) -> Vec<FieldElement<Self::Field>> {
        let first_row = frame.get_row(0);
        let second_row = frame.get_row(1);
        let third_row = frame.get_row(2);

        vec![third_row[0].clone() - second_row[0].clone() - first_row[0].clone()]
    }

    fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
        let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn context(&self) -> air::context::AirContext {
        self.context.clone()
    }
}

pub fn fibonacci_trace<F: IsField>(
    initial_values: [FieldElement<F>; 2],
    trace_length: usize,
) -> Vec<Vec<FieldElement<F>>> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    ret.push(initial_values[0].clone());
    ret.push(initial_values[1].clone());

    for i in 2..(trace_length) {
        ret.push(ret[i - 1].clone() + ret[i - 2].clone());
    }

    vec![ret]
}
