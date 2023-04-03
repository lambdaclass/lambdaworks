//use lambdaworks_math::field::fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField;
use lambdaworks_stark::PrimeField;
use lambdaworks_stark::ProofConfig;
use lambdaworks_stark::StarkProof;
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

    pub fn __neg__(&self) -> Self {
        Self(-&self.0)
    }

    pub fn value(&self) -> PyU256 {
        PyU256(*self.0.value())
    }

    pub fn inv(&self) -> Self {
        Self(self.0.inv())
    }

    pub fn pow(&self, pyexp: &PyInt) -> PyResult<Self> {
        let exp: u64 = pyexp.extract()?;
        Ok(Self(self.0.pow(exp)))
    }

    pub fn __pow__(&self, pyexp: &PyInt, modulo: Option<&PyInt>) -> PyResult<Py<PyAny>> {
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
pub struct PyStarkProof(StarkProof<PrimeField>);

// #[pyfunction]
// pub fn prove(trace: &PyList, proof_config: &PyProofConfig) -> PyResult<PyStarkProof> {
//     // FIXME is there a better way of taking a list of FieldElements as parameters?
//     let trace = {
//         let mut v: Vec<FE> = Vec::with_capacity(trace.len());
//         for pyelem in trace {
//             let fe = PyFieldElement::extract(pyelem)?;
//             v.push(fe.0);
//         }
//         v
//     };

//     Ok(PyStarkProof(lambdaworks_stark::prover::prove(
//         &trace,
//         &proof_config.0,
//     )))
// }

#[pyfunction]
pub fn verify(_stark_proof: &PyStarkProof) -> bool {
    //lambdaworks_stark::verifier::verify(&stark_proof.0)
    true
}
