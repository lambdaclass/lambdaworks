use self::{
    constraints::boundary::BoundaryConstraints, context::AirContext, frame::Frame,
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

pub trait AIR<F: IsField> {
    fn new(trace: TraceTable, context: AirContext) -> Self;
    fn compute_transition(&self, frame: &Frame<F>) -> Vec<FieldElement<F>>;
    fn compute_boundary_constraints(&self) -> BoundaryConstraints<FieldElement<F>>;
    fn transition_divisors(&self) -> Vec<Polynomial<FieldElement<F>>>;
    fn context(&self) -> AirContext;
}
