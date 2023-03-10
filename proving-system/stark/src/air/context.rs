pub struct AirContext {
    trace_info: (usize, usize),
    transition_degrees: Vec<usize>,
    num_assertions: usize,
    num_transition_constraints: usize,
}
