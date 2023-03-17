use pyo3::prelude::*;

#[pyclass(name = "Proof")]
pub struct PyProof {
    pub merkle_path: Vec<FieldElement<F>>,
    pub hasher: H,
}
