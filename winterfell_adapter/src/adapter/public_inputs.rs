use crate::field_element::element::AdapterFieldElement;
use winter_air::{
    Air, AirContext, Assertion, AuxTraceRandElements, EvaluationFrame,
    ProofOptions as WinterProofOptions, TraceInfo, TransitionConstraintDegree,
};
use winter_prover::Trace;

#[derive(Clone)]
pub struct AirAdapterPublicInputs<A, T, FE>
where
    A: Air,
    A::PublicInputs: Clone,
    T: Trace<BaseField = FE> + Clone,
{
    pub(crate) winterfell_public_inputs: A::PublicInputs,
    pub(crate) transition_degrees: Vec<usize>,
    pub(crate) transition_exemptions: Vec<usize>,
    pub(crate) transition_offsets: Vec<usize>,
    pub(crate) trace: T,
    pub(crate) trace_info: TraceInfo,
}
