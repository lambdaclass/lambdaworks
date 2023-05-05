use crate::{
    air::{
        self,
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        context::AirContext,
        AIR,
    },
    fri::FieldElement,
};
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

    fn compute_transition(
        &self,
        frame: &air::frame::Frame<Self::Field>,
    ) -> Vec<FieldElement<Self::Field>> {
        let first_row = frame.get_row(0);
        let second_row = frame.get_row(1);
        let third_row = frame.get_row(2);

        vec![third_row[0] - second_row[0] - first_row[0]]
    }

    fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
        let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());
        let result = BoundaryConstraint::new_simple(3, FieldElement::<Self::Field>::from(3));

        BoundaryConstraints::from_constraints(vec![a0, a1, result])
    }

    fn context(&self) -> air::context::AirContext {
        self.context.clone()
    }
}
