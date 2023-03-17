use lambdaworks_stark::FE;

use pyo3::*;

use crate::math::unsigned_integer::element::PyU256;

#[derive(Clone)]
pub struct FieldElement(pub FE);

#[pymethods]
impl FieldElement {
    #[new]
    pub fn new(value: &PyU256) -> Self {
        Self(FE::new(value.0))
    }

    fn __add__(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }
}
