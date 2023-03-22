use lambdaworks_math::field::{element::FieldElement, traits::IsField};

#[derive(Clone)]
pub struct TraceTable<F: IsField> {
    pub table: Vec<FieldElement<F>>,
    pub num_cols: usize,
}

impl<F: IsField> TraceTable<F> {
    pub fn new(table: Vec<FieldElement<F>>, num_cols: usize) -> Self {
        Self { table, num_cols }
    }
}
