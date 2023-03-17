use pyo3::{types::PyModule, *};

pub mod element;

pub fn module(py: Python) -> PyResult<&PyModule> {
    let m = PyModule::new(py, "unsigned_integer")?;
    m.add_submodule(element::module(py)?)?;
    Ok(m)
}
