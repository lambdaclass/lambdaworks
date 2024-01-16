use super::boundary::BoundaryConstraints;
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
use crate::trace::LDETraceTable;
use crate::traits::AIR;
use crate::{frame::Frame, prover::evaluate_polynomial_on_lde_domain};
use itertools::Itertools;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::{fft::errors::FFTError, field::element::FieldElement, traits::AsBytes};
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

pub struct ConstraintEvaluator<A: AIR> {
    boundary_constraints: BoundaryConstraints<A::FieldExtension>,
}
impl<A: AIR> ConstraintEvaluator<A> {
    pub fn new(air: &A, rap_challenges: &[FieldElement<A::FieldExtension>]) -> Self {
        let boundary_constraints = air.boundary_constraints(rap_challenges);

        Self {
            boundary_constraints,
        }
    }

    pub(crate) fn evaluate(
        &self,
        air: &A,
        lde_trace: &LDETraceTable<A::Field, A::FieldExtension>,
        domain: &Domain<A::Field>,
        transition_coefficients: &[FieldElement<A::FieldExtension>],
        boundary_coefficients: &[FieldElement<A::FieldExtension>],
        rap_challenges: &[FieldElement<A::FieldExtension>],
    ) -> Vec<FieldElement<A::FieldExtension>>
    where
        FieldElement<A::Field>: AsBytes + Send + Sync,
        FieldElement<A::FieldExtension>: AsBytes + Send + Sync,
        A: Send + Sync,
    {
        let boundary_constraints = &self.boundary_constraints;
        let number_of_b_constraints = boundary_constraints.constraints.len();
        let boundary_zerofiers_inverse_evaluations: Vec<Vec<FieldElement<A::Field>>> =
            boundary_constraints
                .constraints
                .iter()
                .map(|bc| {
                    let point = &domain.trace_primitive_root.pow(bc.step as u64);
                    let mut evals = domain
                        .lde_roots_of_unity_coset
                        .iter()
                        .map(|v| v.clone() - point)
                        .collect::<Vec<FieldElement<A::Field>>>();
                    FieldElement::inplace_batch_inverse(&mut evals).unwrap();
                    evals
                })
                .collect::<Vec<Vec<FieldElement<A::Field>>>>();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_polys: Vec<Polynomial<FieldElement<A::Field>>> = Vec::new();

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
            .collect::<Result<Vec<Vec<FieldElement<A::Field>>>, FFTError>>()
            .unwrap();

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

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_zerofiers = Vec::new();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        check_boundary_polys_divisibility(boundary_polys, boundary_zerofiers);

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let mut transition_evaluations = Vec::new();

        let mut transition_zerofiers_evals = air.transition_zerofier_evaluations(domain);

        // Iterate over all LDE domain and compute the part of the composition polynomial
        // related to the transition constraints and add it to the already computed part of the
        // boundary constraints.
        let evaluations_t_iter = 0..domain.lde_roots_of_unity_coset.len();
        let evaluations_t = evaluations_t_iter
            .zip(&boundary_evaluation)
            .map(|(i, boundary)| {
                let frame = Frame::read_from_lde(lde_trace, i, &air.context().transition_offsets);

                let periodic_values: Vec<_> = lde_periodic_columns
                    .iter()
                    .map(|col| col[i].clone())
                    .collect();

                // Compute all the transition constraints at this point of the LDE domain.
                let evaluations_transition =
                    air.compute_transition_prover(&frame, &periodic_values, rap_challenges);

                #[cfg(all(debug_assertions, not(feature = "parallel")))]
                transition_evaluations.push(evaluations_transition.clone());

                let transition_zerofiers_eval = transition_zerofiers_evals.next().unwrap();

                // Add each term of the transition constraints to the composition polynomial, including the zerofier,
                // the challenge and the exemption polynomial if it is necessary.
                let acc_transition = itertools::izip!(
                    evaluations_transition,
                    transition_zerofiers_eval,
                    transition_coefficients
                )
                .fold(FieldElement::zero(), |acc, (eval, zerof_eval, beta)| {
                    acc + zerof_eval * eval * beta
                });

                acc_transition + boundary
            })
            .collect();

        evaluations_t
    }
}
