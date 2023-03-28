use crate::crypto::merkle_tree::proof::PyU64Proof;
use lambdaworks_crypto::merkle_tree::{MerkleTreeDefault, U64MerkleTree, U64FE};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
use lambdaworks_math::field::element::FieldElement;
use pyo3::{
    prelude::*,
    types::{PyInt, PyList},
};

#[pyclass(name = "BLS12381PrimeField")]
#[derive(Clone)]
pub struct PyBLS12381PrimeFieldElement(FieldElement<BLS12381PrimeField>);

#[pyclass(name = "U64FE")]
#[derive(Clone)]
pub struct PyU64FE(U64FE);

impl From<PyU64FE> for U64FE {
    fn from(py_u64fe: PyU64FE) -> Self {
        py_u64fe.0
    }
}

#[pymethods]
impl PyU64FE {
    #[new]
    pub fn new(value: &PyInt) -> PyResult<Self> {
        let a = U64FE::new(PyAny::extract(&value)?);

        Ok(PyU64FE(a))
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

    fn __floordiv__(&self, other: &Self) -> Self {
        Self(&self.0 / &other.0)
    }
}

#[pyclass(name = "U64MerkleTree")]
pub struct PyU64MerkleTree(U64MerkleTree);

#[pymethods]
impl PyU64MerkleTree {
    #[new]
    pub fn new(values: &PyList) -> PyResult<Self> {
        let values = {
            let mut v: Vec<U64FE> = Vec::with_capacity(values.len());
            for pyelem in values {
                let rsvalue = PyU64FE::extract(pyelem)?;
                v.push(rsvalue.0);
            }
            v
        };
        Ok(PyU64MerkleTree(U64MerkleTree::build(values.as_slice())))
    }

    pub fn get_proof(&self, value: &PyU64FE) -> Option<PyU64Proof> {
        let x = self.0.get_proof(&value.0)?;

        Some(PyU64Proof::from(x))
    }

    pub fn get_proof_by_pos(&self, pos: usize) -> Option<PyU64Proof> {
        let proof = self.0.get_proof_by_pos(pos)?;

        Some(PyU64Proof::from(proof))
    }
}

#[pyclass(name = "MerkleTreeDefault")]
pub struct PyMerkleTreeDefault(MerkleTreeDefault);

#[pymethods]
impl PyMerkleTreeDefault {
    #[new]
    pub fn new(values: &PyList) -> PyResult<Self> {
        let values = {
            let mut v: Vec<FieldElement<BLS12381PrimeField>> = Vec::with_capacity(values.len());
            for pyelem in values {
                let rsvalue = PyBLS12381PrimeFieldElement::extract(pyelem)?;
                v.push(rsvalue.0);
            }
            v
        };
        Ok(PyMerkleTreeDefault(MerkleTreeDefault::build(
            values.as_slice(),
        )))
    }

    // pub fn get_proof(&self, value: &PyBLS12381PrimeFieldElement) -> Option<PyBLS12381Proof> {
    //     let x = self.0.get_proof(&value.0)?;

    //     Some(PyBLS12381Proof::from(x))
    // }

    // pub fn get_proof_by_pos(&self, pos: usize) -> Option<PyBLS12381Proof>{
    //     let proof = self.0.get_proof_by_pos(pos)?;

    //     Some(PyBLS12381Proof::from(proof))
    //}
}
