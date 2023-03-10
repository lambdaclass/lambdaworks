use crate::FE;

use self::{constraints::boundary::BoundaryConstraints, context::AirContext, trace::TraceTable};
use crate::air::frame::EvaluationFrame;

pub mod constraints;
pub mod context;
pub mod frame;
pub mod trace;

pub trait Air {
    type Frame: EvaluationFrame;

    fn new(trace: TraceTable, context: AirContext) -> Self;

    fn boundary_constraints(&self) -> BoundaryConstraints<FE>;

    fn compute_transition(&self, frame: Self::Frame) -> Vec<FE>;

    fn context(&self) -> AirContext;
}
