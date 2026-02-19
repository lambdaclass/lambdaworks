//! # AIR (Algebraic Intermediate Representation) Traits
//!
//! This module defines the core traits for expressing computations as algebraic constraints
//! over an execution trace.
//!
//! ## What is AIR?
//!
//! AIR is a way to represent computations as polynomial constraints that must be satisfied
//! by a valid execution trace. Given a computation with `n` steps and `w` state variables,
//! the execution trace is a `n × w` matrix where each row represents the state at one step.
//!
//! ## Constraint Types
//!
//! ### Boundary Constraints
//! Fix specific values at specific positions in the trace:
//! ```text
//! B_i(x) = t_col(x) - value,  where x = ω^step
//! ```
//!
//! ### Transition Constraints
//! Relate values in consecutive rows:
//! ```text
//! C_i(t_0(x), t_1(x), ..., t_0(ω·x), t_1(ω·x), ...) = 0
//! ```
//! where `t_j(x)` are trace column polynomials and `ω` is the trace domain generator.
//!
//! ## Composition Polynomial
//!
//! All constraints are combined into a single composition polynomial:
//! ```text
//! H(x) = Σ α_i · C_i(x) / Z_i(x)
//! ```
//! where:
//! - `C_i(x)` are the constraint polynomials
//! - `Z_i(x)` are the zerofier polynomials (vanish where constraints should hold)
//! - `α_i` are random coefficients from the verifier
//!
//! ## Zerofier Polynomials
//!
//! Zerofiers define where constraints must be satisfied:
//! - **Full zerofier**: `Z(x) = x^n - 1` (all rows)
//! - **Boundary zerofier**: `(x - ω^step)` (single row)
//! - **Periodic zerofier**: `x^(n/period) - 1` (every `period` rows)
//!
//! ## References
//!
//! - [STARK Paper](https://eprint.iacr.org/2018/046) Section 4 - AIR formalization
//! - [ethSTARK Documentation](https://eprint.iacr.org/2021/582) - Practical AIR design
//! - [Cairo Whitepaper](https://eprint.iacr.org/2021/1063) - AIR for CPU execution

use std::collections::HashMap;
use std::ops::Div;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
    polynomial::Polynomial,
};

use crate::{constraints::transition::TransitionConstraint, domain::Domain, prover::ProvingError};

use super::{
    constraints::boundary::BoundaryConstraints, context::AirContext, frame::Frame,
    proof::options::ProofOptions, trace::TraceTable,
};

/// Key for grouping constraints with identical zerofier parameters.
/// Components: (period, offset, exemptions_period, periodic_exemptions_offset, end_exemptions)
type ZerofierGroupKey = (usize, usize, Option<usize>, Option<usize>, usize);

/// Key for caching base zerofier evaluations (without end exemptions).
/// Components: (period, offset, exemptions_period, periodic_exemptions_offset)
type BaseZerofierKey = (usize, usize, Option<usize>, Option<usize>);

/// Key for caching end exemptions polynomial evaluations.
/// Components: (end_exemptions, period)
type EndExemptionsKey = (usize, usize);

/// Compute base zerofier evaluations (without end exemptions)
#[allow(clippy::too_many_arguments)]
fn compute_base_zerofier<F: IsFFTField>(
    period: usize,
    offset: usize,
    exemptions_period: Option<usize>,
    periodic_exemptions_offset: Option<usize>,
    blowup_factor: usize,
    trace_length: usize,
    trace_primitive_root: &FieldElement<F>,
    coset_offset: &FieldElement<F>,
    lde_root: &FieldElement<F>,
) -> Vec<FieldElement<F>> {
    use itertools::Itertools;

    if let Some(exemptions_period_val) = exemptions_period {
        let last_exponent = blowup_factor * exemptions_period_val;
        (0..last_exponent)
            .map(|exponent| {
                let x = lde_root.pow(exponent);
                let offset_times_x = coset_offset * &x;
                let offset_exponent = trace_length
                    * periodic_exemptions_offset.expect(
                        "periodic_exemptions_offset must be Some when exemptions_period is Some",
                    )
                    / exemptions_period_val;

                let numerator = offset_times_x.pow(trace_length / exemptions_period_val)
                    - trace_primitive_root.pow(offset_exponent);
                let denominator = offset_times_x.pow(trace_length / period)
                    - trace_primitive_root.pow(offset * trace_length / period);

                // Safety: The denominator is non-zero because the coset offset ensures
                // lde_root powers are disjoint from trace_primitive_root powers
                numerator.div(denominator).expect(
                    "zerofier denominator should be non-zero: coset offset ensures disjoint domains"
                )
            })
            .collect()
    } else {
        // Standard case: compute 1/(x^(n/period) - g^(offset*n/period))
        let last_exponent = blowup_factor * period;
        let mut evaluations = (0..last_exponent)
            .map(|exponent| {
                let x = lde_root.pow(exponent);
                (coset_offset * &x).pow(trace_length / period)
                    - trace_primitive_root.pow(offset * trace_length / period)
            })
            .collect_vec();

        FieldElement::inplace_batch_inverse(&mut evaluations)
            .expect("batch inverse failed: zerofier evaluation contains zero element");
        evaluations
    }
}

/// Compute end exemptions polynomial evaluations
fn compute_end_exemptions_evals<F: IsFFTField>(
    end_exemptions: usize,
    period: usize,
    blowup_factor: usize,
    trace_length: usize,
    trace_primitive_root: &FieldElement<F>,
    coset_offset: &FieldElement<F>,
    interpolation_domain_size: usize,
) -> Vec<FieldElement<F>> {
    use crate::prover::evaluate_polynomial_on_lde_domain;

    let end_exemptions_poly =
        compute_end_exemptions_poly(end_exemptions, period, trace_primitive_root, trace_length);
    evaluate_polynomial_on_lde_domain(
        &end_exemptions_poly,
        blowup_factor,
        interpolation_domain_size,
        coset_offset,
    )
    .expect("failed to evaluate end exemptions polynomial on LDE domain")
}

/// Compute the end exemptions polynomial
fn compute_end_exemptions_poly<F: IsFFTField>(
    end_exemptions: usize,
    period: usize,
    trace_primitive_root: &FieldElement<F>,
    trace_length: usize,
) -> Polynomial<FieldElement<F>> {
    let one_poly = Polynomial::new_monomial(FieldElement::<F>::one(), 0);
    if end_exemptions == 0 {
        return one_poly;
    }
    (1..=end_exemptions)
        .map(|exemption| trace_primitive_root.pow(trace_length - exemption * period))
        .fold(one_poly, |acc, offset| {
            acc * (Polynomial::new_monomial(FieldElement::<F>::one(), 1) - offset)
        })
}

/// This enum is necessary because, while both the prover and verifier perform the same operations
///  to compute transition constraints, their frames differ.
///  The prover uses a frame containing elements from both the base field and its extension
/// (common when working with small fields and challengers in the extension).
/// In contrast, the verifier, lacking access to the trace and relying solely on evaluations at the challengers,
/// works with a frame that contains only elements from the extension.
pub enum TransitionEvaluationContext<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    Prover {
        frame: &'a Frame<'a, F, E>,
        periodic_values: &'a [FieldElement<F>],
        rap_challenges: &'a [FieldElement<E>],
    },
    Verifier {
        frame: &'a Frame<'a, E, E>,
        periodic_values: &'a [FieldElement<E>],
        rap_challenges: &'a [FieldElement<E>],
    },
}

impl<'a, F, E> TransitionEvaluationContext<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    pub fn new_prover(
        frame: &'a Frame<'a, F, E>,
        periodic_values: &'a [FieldElement<F>],
        rap_challenges: &'a [FieldElement<E>],
    ) -> Self {
        Self::Prover {
            frame,
            periodic_values,
            rap_challenges,
        }
    }

    pub fn new_verifier(
        frame: &'a Frame<'a, E, E>,
        periodic_values: &'a [FieldElement<E>],
        rap_challenges: &'a [FieldElement<E>],
    ) -> Self {
        Self::Verifier {
            frame,
            periodic_values,
            rap_challenges,
        }
    }
}

/// AIR is a representation of the Constraints
pub trait AIR: Send + Sync {
    type Field: IsFFTField + IsSubFieldOf<Self::FieldExtension> + Send + Sync;
    type FieldExtension: IsField + Send + Sync;
    type PublicInputs;

    fn step_size(&self) -> usize;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self
    where
        Self: Sized;

    fn build_auxiliary_trace(
        &self,
        _main_trace: &mut TraceTable<Self::Field, Self::FieldExtension>,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> Result<(), ProvingError> {
        Ok(())
    }

    fn build_rap_challenges(
        &self,
        _transcript: &mut dyn IsStarkTranscript<Self::FieldExtension, Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        Vec::new()
    }

    /// Returns the amount main trace columns and auxiliary trace columns
    fn trace_layout(&self) -> (usize, usize);

    fn has_trace_interaction(&self) -> bool {
        let (_main_trace_columns, aux_trace_columns) = self.trace_layout();
        aux_trace_columns != 0
    }

    fn num_auxiliary_rap_columns(&self) -> usize {
        self.trace_layout().1
    }

    fn composition_poly_degree_bound(&self) -> usize;

    /// The method called by the prover to evaluate the transitions corresponding to an evaluation frame.
    /// In the case of the prover, the main evaluation table of the frame takes values in
    /// `Self::Field`, since they are the evaluations of the main trace at the LDE domain.
    /// In the case of the verifier, the frame take elements of Self::FieldExtension.
    fn compute_transition(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        let mut evaluations =
            vec![FieldElement::<Self::FieldExtension>::zero(); self.num_transition_constraints()];
        self.compute_transition_into(evaluation_context, &mut evaluations);
        evaluations
    }

    /// Evaluate transition constraints into a pre-allocated buffer.
    /// This avoids allocation when called in a loop with a reusable buffer.
    /// The buffer must have length >= `num_transition_constraints()`.
    fn compute_transition_into(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
        evaluations: &mut [FieldElement<Self::FieldExtension>],
    ) {
        // Zero out the buffer
        for eval in evaluations.iter_mut() {
            *eval = FieldElement::zero();
        }
        self.transition_constraints()
            .iter()
            .for_each(|c| c.evaluate(evaluation_context, evaluations));
    }

    fn boundary_constraints(
        &self,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension>;

    fn context(&self) -> &AirContext;

    fn trace_length(&self) -> usize;

    fn options(&self) -> &ProofOptions {
        &self.context().proof_options
    }

    fn blowup_factor(&self) -> u8 {
        self.options().blowup_factor
    }

    fn coset_offset(&self) -> FieldElement<Self::Field> {
        FieldElement::from(self.options().coset_offset)
    }

    fn trace_primitive_root(&self) -> FieldElement<Self::Field> {
        let trace_length = self.trace_length();
        let root_of_unity_order = u64::from(trace_length.trailing_zeros());

        Self::Field::get_primitive_root_of_unity(root_of_unity_order).expect(
            "failed to get primitive root of unity: trace length may exceed field's two-adicity",
        )
    }

    fn num_transition_constraints(&self) -> usize {
        self.context().num_transition_constraints
    }

    fn pub_inputs(&self) -> &Self::PublicInputs;

    fn get_periodic_column_values(&self) -> Vec<Vec<FieldElement<Self::Field>>> {
        vec![]
    }

    fn get_periodic_column_polynomials(&self) -> Vec<Polynomial<FieldElement<Self::Field>>> {
        let mut result = Vec::new();
        for periodic_column in self.get_periodic_column_values() {
            let values: Vec<_> = periodic_column
                .iter()
                .cycle()
                .take(self.trace_length())
                .cloned()
                .collect();
            let poly =
                Polynomial::<FieldElement<Self::Field>>::interpolate_fft::<Self::Field>(&values)
                    .expect("FFT interpolation of periodic column must succeed");
            result.push(poly);
        }
        result
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>>;

    fn transition_zerofier_evaluations(
        &self,
        domain: &Domain<Self::Field>,
    ) -> Vec<Vec<FieldElement<Self::Field>>> {
        #[cfg(feature = "parallel")]
        use rayon::prelude::*;

        let constraints = self.transition_constraints();
        let blowup_factor = domain.blowup_factor;
        let trace_length = domain.trace_roots_of_unity.len();
        let trace_primitive_root = &domain.trace_primitive_root;
        let coset_offset = &domain.coset_offset;
        let lde_root_order = u64::from((blowup_factor * trace_length).trailing_zeros());
        let lde_root = Self::Field::get_primitive_root_of_unity(lde_root_order)
            .expect("primitive root of unity must exist for LDE domain size");

        // Step 1: Collect unique keys
        let mut unique_base_keys: Vec<BaseZerofierKey> = Vec::new();
        let mut unique_end_exemptions_keys: Vec<EndExemptionsKey> = Vec::new();

        for c in constraints.iter() {
            let base_key: BaseZerofierKey = (
                c.period(),
                c.offset(),
                c.exemptions_period(),
                c.periodic_exemptions_offset(),
            );
            if !unique_base_keys.contains(&base_key) {
                unique_base_keys.push(base_key);
            }

            let end_key: EndExemptionsKey = (c.end_exemptions(), c.period());
            if !unique_end_exemptions_keys.contains(&end_key) {
                unique_end_exemptions_keys.push(end_key);
            }
        }

        // Step 2: Compute base zerofiers (parallel if feature enabled)
        #[cfg(feature = "parallel")]
        let base_zerofiers: Vec<_> = unique_base_keys
            .par_iter()
            .map(
                |&(period, offset, exemptions_period, periodic_exemptions_offset)| {
                    compute_base_zerofier(
                        period,
                        offset,
                        exemptions_period,
                        periodic_exemptions_offset,
                        blowup_factor,
                        trace_length,
                        trace_primitive_root,
                        coset_offset,
                        &lde_root,
                    )
                },
            )
            .collect();

        #[cfg(not(feature = "parallel"))]
        let base_zerofiers: Vec<_> = unique_base_keys
            .iter()
            .map(
                |&(period, offset, exemptions_period, periodic_exemptions_offset)| {
                    compute_base_zerofier(
                        period,
                        offset,
                        exemptions_period,
                        periodic_exemptions_offset,
                        blowup_factor,
                        trace_length,
                        trace_primitive_root,
                        coset_offset,
                        &lde_root,
                    )
                },
            )
            .collect();

        let base_zerofier_map: HashMap<BaseZerofierKey, Vec<FieldElement<Self::Field>>> =
            unique_base_keys.into_iter().zip(base_zerofiers).collect();

        // Step 3: Compute end exemptions polynomial evaluations (parallel if feature enabled)
        #[cfg(feature = "parallel")]
        let end_exemptions_evals: Vec<_> = unique_end_exemptions_keys
            .par_iter()
            .map(|&(end_exemptions, period)| {
                compute_end_exemptions_evals(
                    end_exemptions,
                    period,
                    blowup_factor,
                    trace_length,
                    trace_primitive_root,
                    coset_offset,
                    domain.interpolation_domain_size,
                )
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let end_exemptions_evals: Vec<_> = unique_end_exemptions_keys
            .iter()
            .map(|&(end_exemptions, period)| {
                compute_end_exemptions_evals(
                    end_exemptions,
                    period,
                    blowup_factor,
                    trace_length,
                    trace_primitive_root,
                    coset_offset,
                    domain.interpolation_domain_size,
                )
            })
            .collect();

        let end_exemptions_map: HashMap<EndExemptionsKey, Vec<FieldElement<Self::Field>>> =
            unique_end_exemptions_keys
                .into_iter()
                .zip(end_exemptions_evals)
                .collect();

        // Step 4: Build final zerofiers by combining base + end_exemptions
        let mut evals = vec![Vec::new(); self.num_transition_constraints()];
        let mut full_zerofier_cache: HashMap<ZerofierGroupKey, Vec<FieldElement<Self::Field>>> =
            HashMap::new();

        for c in constraints.iter() {
            let period = c.period();
            let offset = c.offset();
            let exemptions_period = c.exemptions_period();
            let periodic_exemptions_offset = c.periodic_exemptions_offset();
            let end_exemptions = c.end_exemptions();

            let full_key = (
                period,
                offset,
                exemptions_period,
                periodic_exemptions_offset,
                end_exemptions,
            );

            // Check if we already have the full zerofier cached
            if let Some(cached) = full_zerofier_cache.get(&full_key) {
                evals[c.constraint_idx()] = cached.clone();
                continue;
            }

            let base_key: BaseZerofierKey = (
                period,
                offset,
                exemptions_period,
                periodic_exemptions_offset,
            );
            let end_key: EndExemptionsKey = (end_exemptions, period);

            let base_zerofier = base_zerofier_map
                .get(&base_key)
                .expect("base_key was inserted into map in previous step");
            let end_exemptions_evals = end_exemptions_map
                .get(&end_key)
                .expect("end_key was inserted into map in previous step");

            // Combine base zerofier with end exemptions
            let cycled_base = base_zerofier
                .iter()
                .cycle()
                .take(end_exemptions_evals.len());

            let final_zerofier: Vec<_> = std::iter::zip(cycled_base, end_exemptions_evals.iter())
                .map(|(base, exemption)| base * exemption)
                .collect();

            full_zerofier_cache.insert(full_key, final_zerofier.clone());
            evals[c.constraint_idx()] = final_zerofier;
        }

        evals
    }
}
