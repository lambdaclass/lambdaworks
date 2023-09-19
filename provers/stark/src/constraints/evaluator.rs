use itertools::Itertools;
use lambdaworks_math::{
    fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset,
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
    traits::ByteConversion,
};

#[cfg(feature = "parallel")]
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
use crate::frame::Frame;
use crate::prover::evaluate_polynomial_on_lde_domain;
use crate::trace::TraceTable;
use crate::traits::AIR;

use super::{boundary::BoundaryConstraints, evaluation_table::ConstraintEvaluationTable};

pub struct ConstraintEvaluator<F: IsFFTField, A: AIR> {
    air: A,
    boundary_constraints: BoundaryConstraints<F>,
}
impl<F: IsFFTField, A: AIR + AIR<Field = F>> ConstraintEvaluator<F, A> {
    pub fn new(air: &A, rap_challenges: &A::RAPChallenges) -> Self {
        let boundary_constraints = air.boundary_constraints(rap_challenges);

        Self {
            air: air.clone(),
            boundary_constraints,
        }
    }

    pub fn evaluate(
        &self,
        lde_trace: &TraceTable<F>,
        domain: &Domain<F>,
        transition_coefficients: &[FieldElement<F>],
        boundary_coefficients: &[FieldElement<F>],
        rap_challenges: &A::RAPChallenges,
    ) -> ConstraintEvaluationTable<F>
    where
        FieldElement<F>: ByteConversion + Send + Sync,
        A: Send + Sync,
        A::RAPChallenges: Send + Sync,
    {
        // The + 1 is for the boundary constraints column
        let mut evaluation_table = ConstraintEvaluationTable::new(
            self.air.context().num_transition_constraints() + 1,
            &domain.lde_roots_of_unity_coset,
        );
        let boundary_constraints = &self.boundary_constraints;
        let number_of_b_constraints = boundary_constraints.constraints.len();
        let boundary_zerofiers_inverse_evaluations: Vec<Vec<FieldElement<F>>> =
            boundary_constraints
                .constraints
                .iter()
                .map(|bc| {
                    let point = &domain.trace_primitive_root.pow(bc.step as u64);
                    let mut evals = domain
                        .lde_roots_of_unity_coset
                        .iter()
                        .map(|v| v.clone() - point)
                        .collect::<Vec<FieldElement<F>>>();
                    FieldElement::inplace_batch_inverse(&mut evals).unwrap();
                    evals
                })
                .collect::<Vec<Vec<FieldElement<F>>>>();

        let trace_length = self.air.trace_length();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_polys: Vec<Polynomial<FieldElement<F>>> = Vec::new();

        let n_col = lde_trace.n_cols;
        let n_elem = domain.lde_roots_of_unity_coset.len();
        let boundary_polys_evaluations = boundary_constraints
            .constraints
            .iter()
            .map(|constraint| {
                let col = constraint.col;
                lde_trace
                    .table
                    .iter()
                    .skip(col)
                    .step_by(n_col)
                    .take(n_elem)
                    .map(|v| v - &constraint.value)
                    .collect::<Vec<FieldElement<F>>>()
            })
            .collect::<Vec<Vec<FieldElement<F>>>>();

        #[cfg(feature = "parallel")]
        let boundary_eval_iter = (0..domain.lde_roots_of_unity_coset.len()).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let boundary_eval_iter = 0..domain.lde_roots_of_unity_coset.len();

        let boundary_evaluation = boundary_eval_iter
            .map(|i| {
                (0..number_of_b_constraints)
                    .zip(boundary_coefficients)
                    .fold(FieldElement::zero(), |acc, (index, beta)| {
                        acc + &boundary_zerofiers_inverse_evaluations[index][i]
                            * beta
                            * &boundary_polys_evaluations[index][i]
                    })
            })
            .collect::<Vec<FieldElement<F>>>();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_zerofiers = Vec::new();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        check_boundary_polys_divisibility(boundary_polys, boundary_zerofiers);

        let blowup_factor = self.air.blowup_factor();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let mut transition_evaluations = Vec::new();

        let transition_exemptions = self.air.transition_exemptions();

        let transition_exemptions_evaluations =
            evaluate_transition_exemptions(transition_exemptions, domain);
        let num_exemptions = self.air.context().num_transition_exemptions;

        let blowup_factor_order = u64::from(blowup_factor.trailing_zeros());

        let offset = FieldElement::<F>::from(self.air.context().proof_options.coset_offset);
        let offset_pow = offset.pow(trace_length);
        let one = FieldElement::<F>::one();
        let mut zerofier_evaluations = get_powers_of_primitive_root_coset(
            blowup_factor_order,
            blowup_factor as usize,
            &offset_pow,
        )
        .unwrap()
        .iter()
        .map(|v| v - &one)
        .collect::<Vec<_>>();

        FieldElement::inplace_batch_inverse(&mut zerofier_evaluations).unwrap();

        // Iterate over trace and domain and compute transitions
        let evaluations_t_iter;
        let zerofier_iter;
        #[cfg(feature = "parallel")]
        {
            evaluations_t_iter = (0..domain.lde_roots_of_unity_coset.len()).into_par_iter();
            zerofier_iter = evaluations_t_iter
                .clone()
                .map(|i| zerofier_evaluations[i % zerofier_evaluations.len()].clone());
        }
        #[cfg(not(feature = "parallel"))]
        {
            evaluations_t_iter = 0..domain.lde_roots_of_unity_coset.len();
            zerofier_iter = zerofier_evaluations.iter().cycle();
        }

        let evaluations_t = evaluations_t_iter
            .zip(&boundary_evaluation)
            .zip(zerofier_iter)
            .map(|((i, boundary), zerofier)| {
                let frame = Frame::read_from_trace(
                    lde_trace,
                    i,
                    blowup_factor,
                    &self.air.context().transition_offsets,
                );

                let evaluations_transition = self.air.compute_transition(&frame, rap_challenges);

                #[cfg(all(debug_assertions, not(feature = "parallel")))]
                transition_evaluations.push(evaluations_transition.clone());

                let acc_transition = evaluations_transition
                    .iter()
                    .zip(&self.air.context().transition_exemptions)
                    .zip(&self.air.context().transition_degrees)
                    .zip(transition_coefficients)
                    .fold(
                        FieldElement::zero(),
                        |acc, (((eval, exemption), _), beta)| {
                            #[cfg(feature = "parallel")]
                            let zerofier = zerofier.clone();

                            if *exemption == 0 {
                                acc + zerofier * beta * eval
                            } else {
                                //TODO: change how exemptions are indexed!
                                if num_exemptions == 1 {
                                    acc + zerofier
                                        * beta
                                        * eval
                                        * &transition_exemptions_evaluations[0][i]
                                } else {
                                    // This case is not used for Cairo Programs, it can be improved in the future
                                    let vector = &self
                                        .air
                                        .context()
                                        .transition_exemptions
                                        .iter()
                                        .cloned()
                                        .filter(|elem| elem > &0)
                                        .unique_by(|elem| *elem)
                                        .collect::<Vec<usize>>();
                                    let index = vector
                                        .iter()
                                        .position(|elem_2| elem_2 == exemption)
                                        .expect("is there");

                                    acc + zerofier
                                        * beta
                                        * eval
                                        * &transition_exemptions_evaluations[index][i]
                                }
                            }
                        },
                    );
                // TODO: Remove clones

                acc_transition + boundary
            })
            .collect::<Vec<FieldElement<F>>>();

        evaluation_table.evaluations_acc = evaluations_t;

        evaluation_table
    }

    /// Given `evaluations` T_i(x) of the trace polynomial composed with the constraint
    /// polynomial at a certain point `x`, computes the following evaluations and returns them:
    ///
    /// T_i(x) (alpha_i * x^(D - D_i) + beta_i) / (Z_i(x))
    ///
    /// where Z is the zerofier of the `i`-th transition constraint polynomial. In the fibonacci
    /// example, T_i(x) is t(x * g^2) - t(x * g) - t(x).
    ///
    /// This method is called in two different scenarios. The first is when the prover
    /// is building these evaluations to compute the composition and DEEP composition polynomials.
    /// The second one is when the verifier needs to check the consistency between the trace and
    /// the composition polynomial. In that case the `evaluations` are over an *out of domain* frame
    /// (in the fibonacci example they are evaluations on the points `z`, `zg`, `zg^2`).
    ///
    /// # Returns
    ///
    /// Returns the sum of the evaluations computed.
    pub fn compute_constraint_composition_poly_evaluations_sum(
        evaluations: &[FieldElement<F>],
        inverse_denominators: &[FieldElement<F>],
        degree_adjustments: &[FieldElement<F>],
        constraint_coeffs: &[(FieldElement<F>, FieldElement<F>)],
    ) -> FieldElement<F> {
        evaluations
            .iter()
            .zip(degree_adjustments)
            .zip(inverse_denominators)
            .zip(constraint_coeffs)
            .fold(
                FieldElement::<F>::zero(),
                |acc, (((ev, _), inv), (_, beta))| acc + ev * beta * inv,
            )
    }
}

fn evaluate_transition_exemptions<F: IsFFTField>(
    transition_exemptions: Vec<Polynomial<FieldElement<F>>>,
    domain: &Domain<F>,
) -> Vec<Vec<FieldElement<F>>>
where
    FieldElement<F>: Send + Sync,
    Polynomial<FieldElement<F>>: Send + Sync,
{
    #[cfg(feature = "parallel")]
    let exemptions_iter = transition_exemptions.par_iter();
    #[cfg(not(feature = "parallel"))]
    let exemptions_iter = transition_exemptions.iter();

    exemptions_iter
        .map(|exemption| {
            evaluate_polynomial_on_lde_domain(
                exemption,
                domain.blowup_factor,
                domain.interpolation_domain_size,
                &domain.coset_offset,
            )
            .unwrap()
        })
        .collect()
}
