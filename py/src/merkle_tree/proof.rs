use lambdaworks_crypto::merkle_tree::U64Proof;
use pyo3::prelude::*;

#[pyclass(name = "Proof")]
pub struct PyProof(Proof);

{
    pub merkle_path: Vec<FieldElement<F>>,
    pub hasher: H
}
