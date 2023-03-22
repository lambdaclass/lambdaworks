use lambdaworks_crypto::merkle_tree::{U64Proof, U64FE};

use lambdaworks_math::traits::ByteConversion;

use pyo3::types::*;
use pyo3::*;

use crate::math::errors::PyByteConversionError;

// FIXME delete this and use Mfachal's implementation.
#[pyclass(name = "U64FE")]
#[derive(Clone)]
pub struct PyU64FE(U64FE);

#[pyclass(name = "U64Proof")]
struct PyU64Proof(U64Proof);

#[pymethods]
impl PyU64Proof {
    pub fn verify(&self, root_hash: &PyU64FE, index: usize, value: &PyU64FE) -> bool {
        self.0.verify(&root_hash.0, index, &value.0)
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
