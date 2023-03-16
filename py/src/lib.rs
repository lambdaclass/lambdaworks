use lambdaworks_math::unsigned_integer::element::U256 as U256Internal;
use lambdaworks_stark::FE;
use pyo3::prelude::*;

#[pyclass]
pub struct U256(U256Internal);

#[pymethods]
impl U256 {
    #[new]
    pub fn new(value: &str) -> PyResult<Self> {
        Ok(Self(U256Internal::from(value)))
    }
}

#[pyclass]
pub struct FieldElement(FE);

#[pymethods]
impl FieldElement {
    #[new]
    pub fn new(value: &U256) -> Self {
        Self(FE::new(value.0))
    }
}

#[pymodule]
fn lambdaworks_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<U256>()?;
    m.add_class::<FieldElement>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use pyo3::prelude::*;
    use pyo3::Python;

    #[test]
    fn lambdaworks_py_test() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new(py, "lambdaworks_py");
            assert!(crate::lambdaworks_py(py, module.unwrap()).is_ok());
        });
    }
}
