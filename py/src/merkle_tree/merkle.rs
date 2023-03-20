
use lambdaworks_crypto::merkle_tree::{MerkleTreeDefault, U64MerkleTree, U64FE};
use lambdaworks_math::field::{element::FieldElement};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
use pyo3::{prelude::*, types::PyList};

#[pyclass(name="BLS12381PrimeField")]
#[derive(Clone)]
pub struct PyBLS12381PrimeFieldElement(FieldElement<BLS12381PrimeField>);

#[pyclass(name="U64FE")]
#[derive(Clone)]
pub struct PyU64FE(U64FE);

#[pyclass(name="U64MerkleTree")]
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

}

#[pyclass(name="MerkleTreeDefault")]
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
        Ok(PyMerkleTreeDefault(MerkleTreeDefault::build(values.as_slice())))
    }

}
