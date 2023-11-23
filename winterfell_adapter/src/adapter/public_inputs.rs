use winter_air::{Air, TraceInfo};

#[derive(Clone)]
pub struct AirAdapterPublicInputs<A, M>
where
    A: Air,
    A::PublicInputs: Clone,
    M: Clone,
{
    pub(crate) winterfell_public_inputs: A::PublicInputs,
    pub(crate) transition_exemptions: Vec<usize>,
    pub(crate) transition_offsets: Vec<usize>,
    pub(crate) trace_info: TraceInfo,
    pub(crate) metadata: M,
}

impl<A, M> AirAdapterPublicInputs<A, M>
where
    A: Air,
    A::PublicInputs: Clone,
    M: Clone,
{
    pub fn new(
        winterfell_public_inputs: A::PublicInputs,
        transition_exemptions: Vec<usize>,
        transition_offsets: Vec<usize>,
        trace_info: TraceInfo,
        metadata: M,
    ) -> Self {
        Self {
            winterfell_public_inputs,
            transition_exemptions,
            transition_offsets,
            trace_info,
            metadata,
        }
    }
}
