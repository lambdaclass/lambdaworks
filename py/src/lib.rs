mod math;
mod proving_systems;

use crate::math::unsigned_integer::element::PyU256;
use crate::proving_systems::stark::PyFieldElement;

use pyo3::prelude::*;

#[pymodule]
fn lambdaworks_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyU256>()?;
    m.add_class::<PyFieldElement>()?;
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
