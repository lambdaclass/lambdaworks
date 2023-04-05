use lambdaworks_stark::FE;

use pyo3::class::basic::CompareOp;
use pyo3::types::*;
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

    fn __richcmp__(&self, other: PyRef<Self>, op: CompareOp) -> Py<PyAny> {
        let py = other.py();
        match op {
            CompareOp::Eq => (self.0 == other.0).into_py(py),
            CompareOp::Ne => (self.0 != other.0).into_py(py),
            _ => py.NotImplemented(),
        }
    }

    fn __add__(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self(&self.0 - &other.0)
    }

    fn __mul__(&self, other: &Self) -> Self {
        Self(&self.0 * &other.0)
    }

    fn __truediv__(&self, other: &Self) -> Self {
        Self(&self.0 / &other.0)
    }

    fn __neg__(&self) -> Self {
        Self(-&self.0)
    }

    pub fn value(&self) -> PyU256 {
        PyU256(*self.0.value())
    }

    pub fn inv(&self) -> Self {
        Self(self.0.inv())
    }

    fn __pow__(&self, pyexp: &PyInt, modulo: Option<&PyInt>) -> PyResult<Py<PyAny>> {
        let py = pyexp.py();
        let exp: u64 = pyexp.extract()?;
        match modulo {
            None => Ok(Self(self.0.pow(exp)).into_py(py)),
            _ => Ok(py.NotImplemented()),
        }
    }

    #[staticmethod]
    pub fn one() -> Self {
        Self(FE::one())
    }

    #[staticmethod]
    pub fn zero() -> Self {
        Self(FE::zero())
    }
}
