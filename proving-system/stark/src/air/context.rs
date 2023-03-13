pub struct AirContext {
    trace_info: (usize, usize),
    transition_degrees: Vec<usize>,
    num_assertions: usize,
    num_transition_constraints: usize,
}

impl AirContext {
    pub fn num_transition_constraints(&self) -> usize {
        self.num_transition_constraints
    }
}
