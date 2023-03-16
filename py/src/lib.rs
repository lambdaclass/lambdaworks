use lambdaworks_math::unsigned_integer::element::U256 as U256Internal;
use lambdaworks_stark::{ProofConfig as ProofConfigInternal, StarkProof as StarkProofInternal, FE};
use pyo3::{prelude::*, types::PyList};

#[pyclass]
pub struct U256(U256Internal);

#[pymethods]
impl U256 {
    #[new]
    pub fn new(value: &str) -> PyResult<Self> {
        Ok(Self(U256Internal::from(value)))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FieldElement(FE);

#[pymethods]
impl FieldElement {
    #[new]
    pub fn new(value: &U256) -> Self {
        Self(FE::new(value.0))
    }

    fn __add__(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }
}

#[pyclass]
pub struct ProofConfig(ProofConfigInternal);

#[pymethods]
impl ProofConfig {
    #[new]
    pub fn new(count_queries: usize, blowup_factor: usize) -> Self {
        Self(ProofConfigInternal {
            count_queries,
            blowup_factor,
        })
    }
}

#[pyclass]
pub struct StarkProof(StarkProofInternal);

#[pyfunction]
fn prove(trace: &PyList, proof_config: &ProofConfig) -> PyResult<StarkProof> {
    // FIXME is there a better way of taking a list of FieldElements as parameters?
    let trace = {
        let mut v: Vec<FE> = Vec::with_capacity(trace.len());
        for pyelem in trace {
            let fe = FieldElement::extract(pyelem)?;
            v.push(fe.0);
        }
        v
    };

    Ok(StarkProof(lambdaworks_stark::prover::prove(
        &trace,
        &proof_config.0,
    )))
}

#[pyfunction]
fn verify(stark_proof: &StarkProof) -> bool {
    lambdaworks_stark::verifier::verify(&stark_proof.0)
}

#[pymodule]
fn lambdaworks_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<U256>()?;
    m.add_class::<FieldElement>()?;
    m.add_class::<ProofConfig>()?;
    m.add_function(wrap_pyfunction!(prove, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use pyo3::prelude::*;
    use pyo3::Python;

    #[test]
    fn lambdaworks_py_test() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new(py, "lambdaworks_py");
            assert!(crate::lambdaworks_py(py, module.unwrap()).is_ok());
        });
    }
}
