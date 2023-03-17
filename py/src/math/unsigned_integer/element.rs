use lambdaworks_math::unsigned_integer::element::U256;

use pyo3::prelude::*;

#[pyclass(name = "U256")]
pub struct PyU256(pub U256);

#[pymethods]
impl PyU256 {
    #[new]
    pub fn new(value: &str) -> PyResult<Self> {
        Ok(Self(U256::from(value)))
    }
}
