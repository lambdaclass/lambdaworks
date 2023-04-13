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
pub struct QuadraticAIR {
    context: AirContext,
}

impl AIR for QuadraticAIR {
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

        vec![second_row[0].clone() - first_row[0].clone() * first_row[0].clone()]
    }

    fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::from(3));

        BoundaryConstraints::from_constraints(vec![a0])
    }

    fn context(&self) -> air::context::AirContext {
        self.context.clone()
    }
}

pub fn quadratic_trace<F: IsField>(
    initial_value: FieldElement<F>,
    trace_length: usize,
) -> Vec<FieldElement<F>> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    ret.push(initial_value);

    for i in 1..(trace_length) {
        ret.push(ret[i - 1].clone() * ret[i - 1].clone());
    }

    ret
}
