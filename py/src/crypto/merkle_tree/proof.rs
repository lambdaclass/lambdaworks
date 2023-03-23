use lambdaworks_crypto::merkle_tree::{U64Proof, U64FE};

use lambdaworks_math::traits::ByteConversion;

use lambdaworks_stark::DefaultHasher;
use pyo3::pyclass::CompareOp;
use pyo3::types::*;
use pyo3::*;

use crate::math::errors::PyByteConversionError;
use crate::merkle_tree::merkle::PyU64FE;

#[pyclass(name = "U64Proof")]
pub struct PyU64Proof(U64Proof);

impl From<U64Proof> for PyU64Proof {
    fn from(proof: U64Proof) -> Self {
        Self(proof)
    }
}

#[pymethods]
impl PyU64Proof {
    #[new]
    pub fn new(merkle_path: &PyList) -> PyResult<Self> {
        let merkle_path = {
            let mut v: Vec<U64FE> = Vec::with_capacity(merkle_path.len());
            for py_elem in merkle_path {
                v.push(PyU64FE::extract(py_elem)?.into());
            }
            v
        };

        Ok(Self(U64Proof {
            hasher: DefaultHasher,
            merkle_path,
        }))
    }

    pub fn __richcmp__(&self, other: PyRef<Self>, op: CompareOp) -> Py<PyAny> {
        let py = other.py();
        match op {
            CompareOp::Eq => (self.0.merkle_path == other.0.merkle_path).into_py(py),
            CompareOp::Ne => (self.0.merkle_path != other.0.merkle_path).into_py(py),
            _ => py.NotImplemented(),
        }
    }

    /* pub fn __eq__(&self, other: &Self) -> bool { */
    /*     self.0.merkle_path == other.0.merkle_path */
    /* } */

    pub fn verify(&self, root_hash: PyU64FE, index: usize, value: PyU64FE) -> bool {
        self.0.verify(&root_hash.into(), index, &value.into())
    }

    pub fn to_bytes_be(&self, py: Python) -> PyObject {
        PyBytes::new(py, &self.0.to_bytes_be()).into()
    }

    pub fn to_bytes_le(&self, py: Python) -> PyObject {
        PyBytes::new(py, &self.0.to_bytes_le()).into()
    }

    #[staticmethod]
    fn from_bytes_be(bytes: &PyBytes) -> Result<Self, PyByteConversionError> {
        let inner = U64Proof::from_bytes_be(bytes.as_bytes())?;
        Ok(Self(inner))
    }

    #[staticmethod]
    fn from_bytes_le(bytes: &PyBytes) -> Result<Self, PyByteConversionError> {
        let inner = U64Proof::from_bytes_le(bytes.as_bytes())?;
        Ok(Self(inner))
    }
}
