#[derive(Clone)]
pub struct AirContext {
    pub options: ProofOptions,
    pub trace_length: usize,
    pub trace_info: (usize, usize),
    pub transition_degrees: Vec<usize>,
    pub transition_exemptions: Vec<usize>,
    pub num_assertions: usize,
    pub num_transition_constraints: usize,
}

impl AirContext {
    pub fn num_transition_constraints(&self) -> usize {
        self.num_transition_constraints
    }

    pub fn transition_degrees(&self) -> Vec<usize> {
        self.transition_degrees.clone()
    }
}

#[derive(Clone)]
pub struct ProofOptions {
    pub(crate) blowup_factor: u8,
}
