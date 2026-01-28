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

use crate::{constraints::transition::TransitionConstraint, domain::Domain};

use super::{
    constraints::boundary::BoundaryConstraints, context::AirContext, frame::Frame,
    proof::options::ProofOptions, trace::TraceTable,
};

type ZerofierGroupKey = (usize, usize, Option<usize>, Option<usize>, usize);

/// Key for caching base zerofier evaluations (without end exemptions)
/// (period, offset, exemptions_period, periodic_exemptions_offset)
type BaseZerofierKey = (usize, usize, Option<usize>, Option<usize>);

/// Key for caching end exemptions polynomial evaluations
/// (end_exemptions, period)
type EndExemptionsKey = (usize, usize);

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
    ) {
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

        Self::Field::get_primitive_root_of_unity(root_of_unity_order).unwrap()
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
                    .unwrap();
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
        use crate::prover::evaluate_polynomial_on_lde_domain;
        use itertools::Itertools;

        let mut evals = vec![Vec::new(); self.num_transition_constraints()];

        // Cache for base zerofier evaluations (without end exemptions multiplication)
        // Key: (period, offset, exemptions_period, periodic_exemptions_offset)
        let mut base_zerofier_cache: HashMap<BaseZerofierKey, Vec<FieldElement<Self::Field>>> =
            HashMap::new();

        // Cache for end exemptions polynomial evaluations
        // Key: (end_exemptions, period)
        let mut end_exemptions_cache: HashMap<EndExemptionsKey, Vec<FieldElement<Self::Field>>> =
            HashMap::new();

        // Full zerofier cache for the complete key (including end_exemptions)
        let mut full_zerofier_cache: HashMap<ZerofierGroupKey, Vec<FieldElement<Self::Field>>> =
            HashMap::new();

        let blowup_factor = domain.blowup_factor;
        let trace_length = domain.trace_roots_of_unity.len();
        let trace_primitive_root = &domain.trace_primitive_root;
        let coset_offset = &domain.coset_offset;

        self.transition_constraints().iter().for_each(|c| {
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
                return;
            }

            let base_key: BaseZerofierKey = (
                period,
                offset,
                exemptions_period,
                periodic_exemptions_offset,
            );

            // Compute or retrieve base zerofier (without end exemptions)
            let base_zerofier = base_zerofier_cache.entry(base_key).or_insert_with(|| {
                let lde_root_order = u64::from((blowup_factor * trace_length).trailing_zeros());
                let lde_root = Self::Field::get_primitive_root_of_unity(lde_root_order).unwrap();

                // Handle periodic exemptions case
                if let Some(exemptions_period_val) = exemptions_period {
                    let last_exponent = blowup_factor * exemptions_period_val;
                    (0..last_exponent)
                        .map(|exponent| {
                            let x = lde_root.pow(exponent);
                            let offset_times_x = coset_offset * &x;
                            let offset_exponent = trace_length
                                * periodic_exemptions_offset.unwrap()
                                / exemptions_period_val;

                            let numerator = offset_times_x
                                .pow(trace_length / exemptions_period_val)
                                - trace_primitive_root.pow(offset_exponent);
                            let denominator = offset_times_x.pow(trace_length / period)
                                - trace_primitive_root.pow(offset * trace_length / period);

                            unsafe { numerator.div(denominator).unwrap_unchecked() }
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

                    FieldElement::inplace_batch_inverse(&mut evaluations).unwrap();
                    evaluations
                }
            });

            // Compute or retrieve end exemptions polynomial evaluations
            let end_exemptions_key: EndExemptionsKey = (end_exemptions, period);
            let end_exemptions_evals = end_exemptions_cache
                .entry(end_exemptions_key)
                .or_insert_with(|| {
                    let end_exemptions_poly =
                        c.end_exemptions_poly(trace_primitive_root, trace_length);
                    evaluate_polynomial_on_lde_domain(
                        &end_exemptions_poly,
                        blowup_factor,
                        domain.interpolation_domain_size,
                        coset_offset,
                    )
                    .unwrap()
                });

            // Combine base zerofier with end exemptions
            let cycled_base = base_zerofier
                .iter()
                .cycle()
                .take(end_exemptions_evals.len());

            let final_zerofier: Vec<_> = std::iter::zip(cycled_base, end_exemptions_evals.iter())
                .map(|(base, exemption)| base * exemption)
                .collect();

            // Cache the full result
            full_zerofier_cache.insert(full_key, final_zerofier.clone());
            evals[c.constraint_idx()] = final_zerofier;
        });

        evals
    }
}
