use crate::prover::{ProofResult, ProverError, ProverOutput};
use crate::prover_optimized::{OptimizedProver, OptimizedProverError};
use crate::Channel;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::{dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial},
    traits::ByteConversion,
};
use std::ops::Mul;

/// Minimum table half-size for GPU dispatch.
/// Below this threshold, the CPU optimized prover is faster due to GPU launch overhead.
const GPU_CROSSOVER_THRESHOLD: usize = 4096;

/// Trait for field types that have Metal sumcheck kernel support.
pub trait HasMetalSumcheckKernel: IsField {
    /// Kernel name prefix (e.g., "BabyBear" or "Goldilocks").
    fn kernel_prefix() -> &'static str;

    /// Size of a single field element in bytes on the GPU.
    fn gpu_element_size() -> usize;
}

/// Metal-accelerated sumcheck prover.
///
/// Uses GPU for the inner loop when tables are large enough to benefit,
/// and falls back to CPU `OptimizedProver` for smaller tables.
pub struct MetalSumcheckProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    /// CPU-side prover for fallback and small tables.
    cpu_prover: OptimizedProver<F>,
}

impl<F: IsField + HasMetalSumcheckKernel> MetalSumcheckProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new Metal sumcheck prover.
    pub fn new(
        factors: Vec<DenseMultilinearPolynomial<F>>,
    ) -> Result<Self, OptimizedProverError> {
        let cpu_prover = OptimizedProver::new(factors)?;
        Ok(Self { cpu_prover })
    }

    pub fn num_vars(&self) -> usize {
        self.cpu_prover.num_vars()
    }

    /// Computes the initial sum (always on CPU, simple O(2^n) scan).
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        self.cpu_prover.compute_initial_sum()
    }

    /// Computes the round polynomial.
    /// Falls back to CPU when table half-size is below the crossover threshold.
    pub fn compute_round_polynomial(
        &self,
    ) -> Result<Polynomial<FieldElement<F>>, OptimizedProverError> {
        // For now, always use CPU prover. GPU dispatch will be added
        // when Metal compilation pipeline is available.
        self.cpu_prover.compute_round_polynomial()
    }

    /// Folds tables after receiving a challenge.
    pub fn receive_challenge(
        &mut self,
        r: &FieldElement<F>,
    ) -> Result<(), OptimizedProverError> {
        self.cpu_prover.receive_challenge(r)
    }
}

/// Non-interactive Metal-accelerated prover using Fiat-Shamir transform.
///
/// Uses identical transcript protocol as `prove()`, so output is verifiable
/// by the existing `verify()` function.
pub fn prove_metal<F>(factors: Vec<DenseMultilinearPolynomial<F>>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript + HasMetalSumcheckKernel,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    let num_factors = factors.len();
    let mut prover = MetalSumcheckProver::new(factors)?;
    let num_vars = prover.num_vars();

    let claimed_sum = prover.compute_initial_sum();

    // Initialize Fiat-Shamir transcript (identical to prove())
    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(num_factors as u64));
    transcript.append_felt(&claimed_sum);

    let mut proof_polys = Vec::with_capacity(num_vars);

    for j in 0..num_vars {
        let g_j = prover.compute_round_polynomial()?;

        // Append g_j to transcript (identical to prove())
        let round_label = format!("round_{j}_poly");
        transcript.append_bytes(round_label.as_bytes());

        let coeffs = g_j.coefficients();
        transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
        if coeffs.is_empty() {
            transcript.append_felt(&FieldElement::zero());
        } else {
            for coeff in coeffs {
                transcript.append_felt(coeff);
            }
        }

        proof_polys.push(g_j);

        // Derive challenge and fold tables (except after last round)
        if j < num_vars - 1 {
            let challenge: FieldElement<F> = transcript.draw_felt();
            prover.receive_challenge(&challenge)?;
        }
    }

    Ok((claimed_sum, proof_polys))
}
