pub struct AirContext {
    pub options: ProofOptions,
    trace_info: (usize, usize),
    transition_degrees: Vec<usize>,
    num_assertions: usize,
    num_transition_constraints: usize,
}

impl AirContext {
    pub fn num_transition_constraints(&self) -> usize {
        self.num_transition_constraints
    }

    pub fn transition_degrees(&self) -> Vec<usize> {
        self.transition_degrees.clone()
    }
}

pub struct ProofOptions {
    pub(crate) blowup_factor: u8,
}
