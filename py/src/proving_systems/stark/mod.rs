use lambdaworks_stark::FE;

use pyo3::*;

use crate::math::unsigned_integer::element::PyU256;

#[pyclass(name = "FieldElement")]
#[derive(Clone)]
pub struct PyFieldElement(pub FE);

#[pymethods]
impl PyFieldElement {
    #[new]
    pub fn new(value: &PyU256) -> Self {
        Self(FE::new(value.0))
    }

    fn __add__(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }
}
