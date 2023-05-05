use super::TraceInfo;

#[derive(Clone, Debug)]
pub struct AirContext {
    pub options: ProofOptions,
    pub trace_info: TraceInfo,
    pub transition_degrees: Vec<usize>,
    pub aux_transition_degrees: Option<Vec<usize>>,
    /// This is a vector with the indices of all the rows that constitute
    /// an evaluation frame. Note that, because of how we write all constraints
    /// in one method (`compute_transitions`), this vector needs to include the
    /// offsets that are needed to compute EVERY transition constraint, even if some
    /// constraints don't use all of the indexes in said offsets.
    pub transition_offsets: Vec<usize>,
    pub aux_transition_offsets: Option<Vec<usize>>,
    pub transition_exemptions: Vec<usize>,
    pub num_transition_constraints: usize,
    pub num_aux_transition_constraints: usize,
}

impl AirContext {
    pub fn num_transition_constraints(&self) -> usize {
        self.num_transition_constraints
    }

    pub fn transition_degrees(&self) -> Vec<usize> {
        self.transition_degrees.clone()
    }
}

#[derive(Clone, Debug)]
pub struct ProofOptions {
    pub blowup_factor: u8,
    pub fri_number_of_queries: u8,
    pub coset_offset: u64,
}
