use crate::field_element::element::AdapterFieldElement;
use winterfell::{Air, TraceInfo};

#[derive(Clone)]
pub struct AirAdapterPublicInputs<A>
where
    A: Air<BaseField = AdapterFieldElement>,
    A::PublicInputs: Clone,
{
    pub(crate) winterfell_public_inputs: A::PublicInputs,
    pub(crate) transition_exemptions: Vec<usize>,
    pub(crate) transition_offsets: Vec<usize>,
    pub(crate) trace_info: TraceInfo,
    pub(crate) composition_poly_degree_bound: usize,
}
