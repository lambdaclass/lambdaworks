use lambdaworks_stark::ProofConfig;
use lambdaworks_stark::StarkProof;
use lambdaworks_stark::FE;

use pyo3::types::*;
use pyo3::*;
use pyo3::class::basic::CompareOp;

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

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Ok(false) //TODO
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

    pub fn value(&self) -> PyU256 {
        PyU256(*self.0.value())
    }

    pub fn inv(&self) -> Self {
        Self(self.0.inv())
    }

    pub fn pow(&self, pyexp: &PyU256) -> Self {
        let exp = pyexp.0;
        Self(self.0.pow(exp))
    }

    pub fn one(&self) -> Self {
        Self(FE::one())
    }

    /// Returns the additive neutral element of the field.
    pub fn zero(&self) -> Self {
        Self(FE::zero())
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
