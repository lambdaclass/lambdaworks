use super::boundary::BoundaryConstraints;
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
use crate::lookup::BusPublicInputs;
use crate::trace::LDETraceTable;
use crate::traits::{TransitionEvaluationContext, AIR};
use crate::{frame::Frame, prover::evaluate_polynomial_on_lde_domain};
use itertools::Itertools;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::{fft::errors::FFTError, field::element::FieldElement};
#[cfg(feature = "parallel")]
use rayon::{
    iter::IndexedParallelIterator,
    prelude::{IntoParallelIterator, ParallelIterator},
};

use std::marker::PhantomData;
#[cfg(feature = "instruments")]
use std::time::Instant;

pub struct ConstraintEvaluator<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
> {
    boundary_constraints: BoundaryConstraints<FieldExtension>,
    logup_table_offset: FieldElement<FieldExtension>,
    phantom: PhantomData<(Field, PI)>,
}
impl<Field, FieldExtension, PI> ConstraintEvaluator<Field, FieldExtension, PI>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
{
    pub fn new(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        rap_challenges: &[FieldElement<FieldExtension>],
        bus_public_inputs: Option<&BusPublicInputs<FieldExtension>>,
    ) -> Self {
        let boundary_constraints = air.boundary_constraints(rap_challenges, bus_public_inputs);

        let logup_table_offset = match bus_public_inputs {
            Some(bpi) => {
                let n = FieldElement::<FieldExtension>::from(air.trace_length() as u64);
                &bpi.table_contribution
                    * n.inv()
                        .expect("trace_length is a power-of-2, so it has an inverse")
            }
            None => FieldElement::zero(),
        };

        Self {
            boundary_constraints,
            logup_table_offset,
            phantom: PhantomData::<(Field, PI)> {},
        }
    }

    pub(crate) fn evaluate(
        &self,
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_coefficients: &[FieldElement<FieldExtension>],
        rap_challenges: &[FieldElement<FieldExtension>],
    ) -> Result<Vec<FieldElement<FieldExtension>>, crate::prover::ProvingError> {
        let boundary_constraints = &self.boundary_constraints;

        // Optimization: Cache boundary zerofiers by step.
        // Multiple boundary constraints can apply at the same step (e.g., constraining different
        // columns at step 0). Since the zerofier only depends on the step, we compute each unique
        // zerofier once and reuse it. This avoids redundant batch inversions (which are expensive)
        // when multiple constraints share the same step.
        //
        // The zerofier for step `s` is: 1/(x - g^s) evaluated at each LDE domain point,
        // where g is the trace primitive root of unity.
        use std::collections::HashMap;
        let mut zerofier_cache: HashMap<usize, Vec<FieldElement<Field>>> = HashMap::new();

        for bc in boundary_constraints.constraints.iter() {
            if let std::collections::hash_map::Entry::Vacant(entry) = zerofier_cache.entry(bc.step)
            {
                let point = domain.trace_primitive_root.pow(bc.step as u64);
                let mut evals: Vec<FieldElement<Field>> = domain
                    .lde_roots_of_unity_coset
                    .iter()
                    .map(|v| v - &point)
                    .collect();
                FieldElement::inplace_batch_inverse(&mut evals)
                    .map_err(|_| crate::prover::ProvingError::BatchInversionFailed)?;
                entry.insert(evals);
            }
        }

        // Pre-build Vec of references to avoid HashMap lookup in the hot loop
        let boundary_zerofiers_refs: Vec<&Vec<FieldElement<Field>>> = boundary_constraints
            .constraints
            .iter()
            .map(|bc| {
                zerofier_cache.get(&bc.step).expect(
                    "zerofier cache miss: boundary constraint step not found in precomputed cache",
                )
            })
            .collect();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_polys: Vec<Polynomial<FieldElement<Field>>> = Vec::new();

        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        let lde_periodic_columns = air
            .get_periodic_column_polynomials()
            .iter()
            .map(|poly| {
                evaluate_polynomial_on_lde_domain(
                    poly,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                )
            })
            .collect::<Result<Vec<Vec<FieldElement<Field>>>, FFTError>>()?;

        #[cfg(feature = "instruments")]
        println!(
            "     Evaluating periodic columns on lde: {:#?}",
            timer.elapsed()
        );

        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        let boundary_polys_evaluations: Vec<Vec<FieldElement<FieldExtension>>> =
            boundary_constraints
                .constraints
                .iter()
                .map(|constraint| {
                    if constraint.is_aux {
                        (0..lde_trace.num_rows())
                            .map(|row| {
                                let v = lde_trace.get_aux(row, constraint.col);
                                v - &constraint.value
                            })
                            .collect_vec()
                    } else {
                        (0..lde_trace.num_rows())
                            .map(|row| {
                                let v = lde_trace.get_main(row, constraint.col);
                                v - &constraint.value
                            })
                            .collect_vec()
                    }
                })
                .collect_vec();

        #[cfg(feature = "instruments")]
        println!("     Created boundary polynomials: {:#?}", timer.elapsed());
        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        #[cfg(feature = "parallel")]
        let boundary_eval_iter = (0..domain.lde_roots_of_unity_coset.len()).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let boundary_eval_iter = 0..domain.lde_roots_of_unity_coset.len();

        let boundary_evaluation: Vec<_> = boundary_eval_iter
            .map(|domain_index| {
                itertools::izip!(
                    &boundary_zerofiers_refs,
                    &boundary_polys_evaluations,
                    boundary_coefficients
                )
                .fold(
                    FieldElement::zero(),
                    |acc, (zerofier, boundary_poly, beta)| {
                        acc + &zerofier[domain_index] * beta * &boundary_poly[domain_index]
                    },
                )
            })
            .collect();

        #[cfg(feature = "instruments")]
        println!(
            "     Evaluated boundary polynomials on LDE: {:#?}",
            timer.elapsed()
        );

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_zerofiers = Vec::new();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        check_boundary_polys_divisibility(boundary_polys, boundary_zerofiers);

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let mut transition_evaluations = Vec::new();

        #[cfg(feature = "instruments")]
        let timer = Instant::now();
        let zerofiers_evals = air.transition_zerofier_evaluations(domain);
        #[cfg(feature = "instruments")]
        println!(
            "     Evaluated transition zerofiers: {:#?}",
            timer.elapsed()
        );

        // Iterate over all LDE domain and compute the part of the composition polynomial
        // related to the transition constraints and add it to the already computed part of the
        // boundary constraints.

        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        #[cfg(feature = "parallel")]
        let evaluations_t = {
            let boundary_evaluation = boundary_evaluation.into_par_iter();
            let evaluations_t_iter = (0..domain.lde_roots_of_unity_coset.len()).into_par_iter();

            evaluations_t_iter
                .zip(boundary_evaluation)
                .map(|(i, boundary)| {
                    let frame =
                        Frame::read_from_lde(lde_trace, i, &air.context().transition_offsets);

                    // Collect periodic values for this index
                    let periodic_values: Vec<_> = lde_periodic_columns
                        .iter()
                        .map(|col| col[i].clone())
                        .collect();

                    let transition_evaluation_context = TransitionEvaluationContext::new_prover(
                        &frame,
                        &periodic_values,
                        rap_challenges,
                        &self.logup_table_offset,
                    );
                    let evaluations_transition =
                        air.compute_transition(&transition_evaluation_context);

                    // Accumulate transition constraints
                    let acc_transition = itertools::izip!(
                        evaluations_transition,
                        &zerofiers_evals,
                        transition_coefficients
                    )
                    .fold(
                        FieldElement::zero(),
                        |acc, (eval, zerof_eval, beta)| {
                            let wrapped_idx = i % zerof_eval.len();
                            acc + &zerof_eval[wrapped_idx] * eval * beta
                        },
                    );

                    acc_transition + boundary
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let evaluations_t = {
            // Pre-allocate reusable buffers for the sequential case
            let num_periodic_cols = lde_periodic_columns.len();
            let mut periodic_values_buffer: Vec<FieldElement<Field>> =
                Vec::with_capacity(num_periodic_cols);
            let mut transition_buffer: Vec<FieldElement<FieldExtension>> =
                vec![FieldElement::zero(); air.num_transition_constraints()];

            let mut result = Vec::with_capacity(domain.lde_roots_of_unity_coset.len());

            for (i, boundary) in boundary_evaluation.into_iter().enumerate() {
                let frame = Frame::read_from_lde(lde_trace, i, &air.context().transition_offsets);

                // Reuse periodic values buffer - clear and refill
                periodic_values_buffer.clear();
                for col in &lde_periodic_columns {
                    periodic_values_buffer.push(col[i].clone());
                }

                let transition_evaluation_context = TransitionEvaluationContext::new_prover(
                    &frame,
                    &periodic_values_buffer,
                    rap_challenges,
                    &self.logup_table_offset,
                );

                // Use buffer-reuse variant to avoid allocation
                air.compute_transition_into(&transition_evaluation_context, &mut transition_buffer);

                #[cfg(debug_assertions)]
                transition_evaluations.push(transition_buffer.clone());

                // Accumulate transition constraints
                let acc_transition = itertools::izip!(
                    &transition_buffer,
                    &zerofiers_evals,
                    transition_coefficients
                )
                .fold(FieldElement::zero(), |acc, (eval, zerof_eval, beta)| {
                    let wrapped_idx = i % zerof_eval.len();
                    acc + &zerof_eval[wrapped_idx] * eval * beta
                });

                result.push(acc_transition + boundary);
            }

            result
        };

        #[cfg(feature = "instruments")]
        println!(
            "     Evaluated transitions and accumulated results: {:#?}",
            timer.elapsed()
        );

        Ok(evaluations_t)
    }
}
