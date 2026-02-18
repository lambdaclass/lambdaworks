use crate::Channel;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::{
        dense_multilinear_poly::DenseMultilinearPolynomial, InterpolateError, Polynomial,
    },
    traits::ByteConversion,
};
use std::ops::Mul;
use thiserror::Error;

pub type ProofResult<F> = (FieldElement<F>, Vec<Polynomial<FieldElement<F>>>);
pub type ProverOutput<F> = Result<ProofResult<F>, ProverError>;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("Factor mismatch: {0}")]
    FactorMismatch(String),
    #[error("Polynomial interpolation failed: {0:?}")]
    InterpolationError(InterpolateError),
    #[error("Invalid prover state: {0}")]
    InvalidState(String),
}

impl From<InterpolateError> for ProverError {
    fn from(e: InterpolateError) -> Self {
        ProverError::InterpolationError(e)
    }
}

/// Table-folding prover for the Sum-Check protocol.
///
/// Maintains evaluation tables that are folded in-place each round,
/// yielding O(d^2 * 2^n) total work instead of the naive O(d^2 * n * 2^{2n}).
pub struct Prover<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    num_factors: usize,
    /// Evaluation tables for each factor, folded in-place each round.
    tables: Vec<Vec<FieldElement<F>>>,
    current_round: usize,
}

impl<F: IsField> Prover<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new Prover instance, cloning each factor's evaluations into tables.
    pub fn new(factors: Vec<DenseMultilinearPolynomial<F>>) -> Result<Self, ProverError> {
        if factors.is_empty() {
            return Err(ProverError::FactorMismatch(
                "At least one polynomial factor is required.".to_string(),
            ));
        }
        let num_vars = factors[0].num_vars();
        if factors.iter().any(|p| p.num_vars() != num_vars) {
            return Err(ProverError::FactorMismatch(
                "All factors must have the same number of variables.".to_string(),
            ));
        }
        let num_factors = factors.len();
        let tables = factors.into_iter().map(|f| f.evals().to_vec()).collect();
        Ok(Self {
            num_vars,
            num_factors,
            tables,
            current_round: 0,
        })
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Computes the initial claimed sum by scanning the tables once — O(d * 2^n).
    pub fn compute_initial_sum(&self) -> Result<FieldElement<F>, ProverError> {
        let table_len = self.tables[0].len();
        let sum = (0..table_len)
            .map(|j| {
                self.tables
                    .iter()
                    .fold(FieldElement::<F>::one(), |acc, table| {
                        acc * table[j].clone()
                    })
            })
            .fold(FieldElement::<F>::zero(), |acc, product| acc + product);
        Ok(sum)
    }

    /// Executes one round of the Sum-Check protocol using table folding.
    ///
    /// 1. If `r_prev` is provided, fold all tables in-place.
    /// 2. For each evaluation point t in {0, 1, ..., d}, scan pairs and
    ///    accumulate the product sum.
    /// 3. Interpolate the univariate polynomial from d+1 point evaluations.
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        if self.current_round >= self.num_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed, no more rounds to run.".to_string(),
            ));
        }

        // Fold tables with the previous challenge
        if let Some(r) = r_prev {
            let half = self.tables[0].len() / 2;
            for table in &mut self.tables {
                for j in 0..half {
                    // table[j] = table[j] + r * (table[j + half] - table[j])
                    //           = (1 - r) * table[j] + r * table[j + half]
                    let lo = table[j].clone();
                    let hi = table[j + half].clone();
                    table[j] = lo.clone() + r.clone() * (hi - lo);
                }
                table.truncate(half);
            }
        }

        let half = self.tables[0].len() / 2;
        let num_eval_points = self.num_factors + 1; // degree d = num_factors, need d+1 points

        let mut xs = Vec::with_capacity(num_eval_points);
        let mut ys = Vec::with_capacity(num_eval_points);

        for t in 0..num_eval_points {
            xs.push(FieldElement::from(t as u64));
            let mut sum = FieldElement::<F>::zero();

            for j in 0..half {
                let mut product = FieldElement::<F>::one();
                for table in &self.tables {
                    let lo = &table[j];
                    let hi = &table[j + half];
                    // Evaluate factor at point t: (1 - t) * lo + t * hi = lo + t * (hi - lo)
                    let val = if t == 0 {
                        lo.clone()
                    } else if t == 1 {
                        hi.clone()
                    } else {
                        let t_fe = &xs[t];
                        lo.clone() + t_fe.clone() * (hi.clone() - lo.clone())
                    };
                    product *= val;
                }
                sum += product;
            }

            ys.push(sum);
        }

        self.current_round += 1;
        Polynomial::interpolate(&xs, &ys).map_err(ProverError::from)
    }
}

pub fn prove<F>(factors: Vec<DenseMultilinearPolynomial<F>>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    let num_factors = factors.len();
    let mut prover = Prover::new(factors)?;
    let num_vars = prover.num_vars();
    let claimed_sum = prover.compute_initial_sum()?;

    let mut transcript = DefaultTranscript::<F>::default();
    crate::init_transcript(&mut transcript, num_vars, num_factors, &claimed_sum);

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut current_challenge: Option<FieldElement<F>> = None;

    for j in 0..num_vars {
        let g_j = prover.round(current_challenge.as_ref())?;
        crate::append_round_poly(&mut transcript, j, &g_j);
        proof_polys.push(g_j);

        current_challenge = (j < num_vars - 1).then(|| transcript.draw_felt());
    }

    Ok((claimed_sum, proof_polys))
}
