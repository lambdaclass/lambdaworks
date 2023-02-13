use lambdaworks_math::field::{element::FieldElement, traits::IsField};

pub struct Trace<T: IsField> {
    inner: Vec<Vec<FieldElement<T>>>,
}

impl<T: IsField> Trace<T> {
    pub fn new(columns: Vec<Vec<FieldElement<T>>>) -> Self {}
}
