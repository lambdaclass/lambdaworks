use winter_air::{Air, TraceInfo};
use winter_prover::Trace;

#[derive(Clone)]
pub struct AirAdapterPublicInputs<A, T, FE>
where
    A: Air,
    A::PublicInputs: Clone,
    T: Trace<BaseField = FE> + Clone,
{
    pub(crate) winterfell_public_inputs: A::PublicInputs,
    pub(crate) transition_exemptions: Vec<usize>,
    pub(crate) transition_offsets: Vec<usize>,
    pub(crate) trace: T,
    pub(crate) trace_info: TraceInfo,
}

impl<A, T, FE> AirAdapterPublicInputs<A, T, FE>
where
    A: Air,
    A::PublicInputs: Clone,
    T: Trace<BaseField = FE> + Clone,
{
    pub fn new(
        winterfell_public_inputs: A::PublicInputs,
        transition_exemptions: Vec<usize>,
        transition_offsets: Vec<usize>,
        trace: T,
        trace_info: TraceInfo,
    ) -> Self {
        Self {
            winterfell_public_inputs,
            transition_exemptions,
            transition_offsets,
            trace,
            trace_info,
        }
    }
}
