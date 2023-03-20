use lambdaworks_stark::FE;
use lambdaworks_stark::ProofConfig;
use lambdaworks_stark::StarkProof;

use pyo3::*;
use pyo3::types::*;

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

    fn __add__(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }
}

#[pyclass(name = "StarkProofConfig")]
pub struct PyProofConfig(pub ProofConfig);

#[pymethods]
impl PyProofConfig {
    #[new]
    pub fn new(count_queries: usize, blowup_factor: usize) -> Self {
        Self(ProofConfig {
            count_queries,
            blowup_factor,
        })
    }
}

#[pyclass]
pub struct PyStarkProof(StarkProof);

#[pyfunction]
pub fn prove(trace: &PyList, proof_config: &PyProofConfig) -> PyResult<PyStarkProof> {
    // FIXME is there a better way of taking a list of FieldElements as parameters?
    let trace = {
        let mut v: Vec<FE> = Vec::with_capacity(trace.len());
        for pyelem in trace {
            let fe = PyFieldElement::extract(pyelem)?;
            v.push(fe.0);
        }
        v
    };

    Ok(PyStarkProof(lambdaworks_stark::prover::prove(
        &trace,
        &proof_config.0,
    )))
}

#[pyfunction]
pub fn verify(stark_proof: &PyStarkProof) -> bool {
    lambdaworks_stark::verifier::verify(&stark_proof.0)
}
