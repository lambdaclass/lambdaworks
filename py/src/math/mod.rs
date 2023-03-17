use pyo3::{types::PyModule, *};

pub mod unsigned_integer;

pub fn module(py: Python) -> PyResult<&PyModule> {
    let m = PyModule::new(py, "math")?;
    m.add_submodule(unsigned_integer::module(py)?)?;
    Ok(m)
}
