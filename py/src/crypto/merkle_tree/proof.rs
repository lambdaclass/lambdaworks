use lambdaworks_crypto::merkle_tree::U64Proof;

use lambdaworks_math::traits::ByteConversion;

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
