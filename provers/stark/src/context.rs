use std::collections::HashSet;

use super::proof::options::ProofOptions;

#[derive(Clone, Debug)]
pub struct AirContext {
    pub proof_options: ProofOptions,
    pub trace_columns: usize,

    /// This is a vector with the indices of all the rows that constitute
    /// an evaluation frame. Note that, because of how we write all constraints
    /// in one method (`compute_transitions`), this vector needs to include the
    /// offsets that are needed to compute EVERY transition constraint, even if some
    /// constraints don't use all of the indexes in said offsets.
    pub transition_offsets: Vec<usize>,
    pub transition_exemptions: Vec<usize>,
    pub num_transition_constraints: usize,
}

impl AirContext {
    pub fn num_transition_constraints(&self) -> usize {
        self.num_transition_constraints
    }

    /// Returns the number of non-trivial different
    /// transition exemptions.
    pub fn num_transition_exemptions(&self) -> usize {
        self.transition_exemptions
            .iter()
            .filter(|&x| *x != 0)
            .collect::<HashSet<_>>()
            .len()
    }
}
