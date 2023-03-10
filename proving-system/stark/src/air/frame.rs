use super::trace::TraceTable;

pub trait EvaluationFrame {
    fn new(base_step: usize, trace: TraceTable) -> Self;
    fn offsets() -> &'static [usize];
}
