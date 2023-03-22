use self::{
    constraints::boundary::BoundaryConstraints,
    context::{AirContext, ProofOptions},
    frame::Frame,
    trace::TraceTable,
};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

pub mod constraints;
pub mod context;
pub mod frame;
pub mod trace;

pub trait AIR: Clone {
    type Field: IsField;

    fn new(trace: TraceTable<Self::Field>, context: AirContext) -> Self;
    fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field>;
    fn compute_transition(&self, frame: &Frame<Self::Field>) -> Vec<FieldElement<Self::Field>>;
    fn transition_divisors(&self) -> Vec<Polynomial<FieldElement<Self::Field>>>;
    fn context(&self) -> AirContext;
    fn options(&self) -> ProofOptions {
        self.context().options
    }
    fn blowup_factor(&self) -> u8 {
        self.options().blowup_factor
    }
}
