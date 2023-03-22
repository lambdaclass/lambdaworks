use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::polynomial::Polynomial;

#[derive(Clone, Default, Debug)]
pub struct TraceTable<F: IsField> {
    /// `table` is column oriented trace element description
    pub table: Vec<Vec<FieldElement<F>>>,
}

impl<F: IsField> TraceTable<F> {
    pub fn compute_trace_polys(
        &self,
        trace_roots_of_unity: &[FieldElement<F>],
    ) -> Vec<Polynomial<FieldElement<F>>> {
        self.table
            .iter()
            .map(|e| Polynomial::interpolate(&trace_roots_of_unity, e))
            .collect()
    }
}
