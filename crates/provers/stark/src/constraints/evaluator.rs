use super::boundary::BoundaryConstraints;
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
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
    ) -> Self {
        let boundary_constraints = air.boundary_constraints(rap_challenges);

        Self {
            boundary_constraints,
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
    ) -> Vec<FieldElement<FieldExtension>> {
        let boundary_constraints = &self.boundary_constraints;
        let number_of_b_constraints = boundary_constraints.constraints.len();
        let boundary_zerofiers_inverse_evaluations: Vec<Vec<FieldElement<Field>>> =
            boundary_constraints
                .constraints
                .iter()
                .map(|bc| {
                    let point = &domain.trace_primitive_root.pow(bc.step as u64);
                    let mut evals = domain
                        .lde_roots_of_unity_coset
                        .iter()
                        .map(|v| v.clone() - point)
                        .collect::<Vec<FieldElement<Field>>>();
                    FieldElement::inplace_batch_inverse(&mut evals).unwrap();
                    evals
                })
                .collect::<Vec<Vec<FieldElement<Field>>>>();

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
            .collect::<Result<Vec<Vec<FieldElement<Field>>>, FFTError>>()
            .unwrap();

        #[cfg(feature = "instruments")]
        println!(
            "     Evaluating periodic columns on lde: {:#?}",
            timer.elapsed()
        );

        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        let boundary_polys_evaluations = boundary_constraints
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
                (0..number_of_b_constraints)
                    .zip(boundary_coefficients)
                    .fold(FieldElement::zero(), |acc, (constraint_index, beta)| {
                        acc + &boundary_zerofiers_inverse_evaluations[constraint_index]
                            [domain_index]
                            * beta
                            * &boundary_polys_evaluations[constraint_index][domain_index]
                    })
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

        // Pre-allocate buffer for periodic values to avoid allocation in the hot loop
        let num_periodic_cols = lde_periodic_columns.len();

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

        evaluations_t
    }
}
