use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    helpers,
    polynomial::Polynomial,
};

use crate::air::{example::cairo::PublicInputs, frame::Frame, trace::TraceTable, AIR};
use std::iter::zip;

use super::{boundary::BoundaryConstraints, evaluation_table::ConstraintEvaluationTable};

pub struct ConstraintEvaluator<'poly, F: IsFFTField, A: AIR> {
    air: A,
    boundary_constraints: BoundaryConstraints<F>,
    trace_polys: &'poly [Polynomial<FieldElement<F>>],
    primitive_root: FieldElement<F>,
}

impl<'poly, F: IsFFTField, A: AIR + AIR<Field = F>> ConstraintEvaluator<'poly, F, A> {
    pub fn new(
        air: &A,
        trace_polys: &'poly [Polynomial<FieldElement<F>>],
        primitive_root: &FieldElement<F>,
        public_input: &A::PublicInput,
        rap_challenges: &A::RAPChallenges,
    ) -> Self {
        let boundary_constraints = air.boundary_constraints(rap_challenges, public_input);

        Self {
            air: air.clone(),
            boundary_constraints,
            trace_polys,
            primitive_root: primitive_root.clone(),
        }
    }

    pub fn evaluate(
        &self,
        lde_trace: &TraceTable<F>,
        lde_domain: &[FieldElement<F>],
        alpha_and_beta_transition_coefficients: &[(FieldElement<F>, FieldElement<F>)],
        alpha_and_beta_boundary_coefficients: &[(FieldElement<F>, FieldElement<F>)],
        rap_challenges: &A::RAPChallenges,
    ) -> ConstraintEvaluationTable<F> {
        // The + 1 is for the boundary constraints column
        let mut evaluation_table = ConstraintEvaluationTable::new(
            self.air.context().num_transition_constraints() + 1,
            lde_domain,
        );
        let n_trace_colums = self.trace_polys.len();
        let boundary_constraints = &self.boundary_constraints;

        let domains =
            boundary_constraints.generate_roots_of_unity(&self.primitive_root, n_trace_colums);
        let values = boundary_constraints.values(n_trace_colums);

        let boundary_polys: Vec<Polynomial<FieldElement<F>>> = zip(domains, values)
            .zip(self.trace_polys)
            .map(|((xs, ys), trace_poly)| trace_poly - &Polynomial::interpolate(&xs, &ys))
            .collect();

        let boundary_zerofiers: Vec<Polynomial<FieldElement<F>>> = (0..n_trace_colums)
            .map(|col| {
                self.boundary_constraints
                    .compute_zerofier(&self.primitive_root, col)
            })
            .collect();

        let boundary_polys_max_degree = boundary_polys
            .iter()
            .zip(&boundary_zerofiers)
            .map(|(poly, zerofier)| poly.degree() - zerofier.degree())
            .max()
            .unwrap();

        let transition_degrees = self.air.context().transition_degrees();
        let transition_zerofiers = self.air.transition_divisors();
        let transition_polys_max_degree = self
            .trace_polys
            .iter()
            .zip(&transition_zerofiers)
            .zip(&transition_degrees)
            .map(|((poly, zerofier), degree)| {
                (poly.degree() * degree).saturating_sub(zerofier.degree())
            })
            .max()
            .unwrap();

        let max_degree = std::cmp::max(transition_polys_max_degree, boundary_polys_max_degree);

        let max_degree_power_of_two = helpers::next_power_of_two(max_degree as u64);

        let blowup_factor = self.air.blowup_factor();

        #[cfg(debug_assertions)]
        for (poly, z) in boundary_polys.iter().zip(boundary_zerofiers.iter()) {
            let (_, b) = poly.clone().long_division_with_remainder(z);
            assert_eq!(b, Polynomial::zero());
        }

        let divisors = self.air.transition_divisors();
        // Iterate over trace and domain and compute transitions
        for (i, d) in lde_domain.iter().enumerate() {
            let frame = Frame::read_from_trace(
                lde_trace,
                i,
                blowup_factor,
                &self.air.context().transition_offsets,
            );

            let mut evaluations = self.air.compute_transition(&frame, rap_challenges);
            evaluations = Self::compute_constraint_composition_poly_evaluations(
                &self.air,
                &evaluations,
                &divisors,
                alpha_and_beta_transition_coefficients,
                max_degree_power_of_two,
                d,
            );

            let boundary_evaluation = zip(&boundary_polys, &boundary_zerofiers)
                .enumerate()
                .map(|(index, (boundary_poly, boundary_zerofier))| {
                    let quotient_degree = boundary_poly.degree() - boundary_zerofier.degree();

                    let (boundary_alpha, boundary_beta) =
                        alpha_and_beta_boundary_coefficients[index].clone();

                    (boundary_poly.evaluate(d) / boundary_zerofier.evaluate(d))
                        * (&boundary_alpha
                            * d.pow(max_degree_power_of_two - (quotient_degree as u64))
                            + &boundary_beta)
                })
                .fold(FieldElement::<F>::zero(), |acc, eval| acc + eval);

            evaluations.push(boundary_evaluation);

            evaluation_table.evaluations.push(evaluations);
        }

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
    pub fn compute_constraint_composition_poly_evaluations(
        air: &A,
        evaluations: &[FieldElement<F>],
        divisors: &[Polynomial<FieldElement<F>>],
        constraint_coeffs: &[(FieldElement<F>, FieldElement<F>)],
        max_degree: u64,
        x: &FieldElement<F>,
    ) -> Vec<FieldElement<F>> {
        // TODO: We should get the trace degree in a better way because in some special cases
        // the trace degree may not be exactly the trace length - 1 but a smaller number.
        let trace_degree = air.context().trace_length - 1;
        let transition_degrees = air.context().transition_degrees();

        let mut ret = Vec::new();
        for (((eval, transition_degree), div), (alpha, beta)) in evaluations
            .iter()
            .zip(transition_degrees)
            .zip(divisors)
            .zip(constraint_coeffs)
        {
            let zerofied_eval = eval / div.evaluate(x);
            let zerofied_degree = trace_degree * transition_degree - div.degree();
            let result =
                zerofied_eval * (alpha * x.pow(max_degree - (zerofied_degree as u64)) + beta);
            ret.push(result);
        }

        ret
    }
}
