use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

use crate::{
    air::{frame::Frame, trace::TraceTable, AIR},
    FE,
};

use super::{boundary::BoundaryConstraints, evaluation_table::ConstraintEvaluationTable};

pub struct ConstraintEvaluator<'a, A: AIR<F>, F: IsField> {
    air: &'a A,
    boundary_constraints: BoundaryConstraints<F>,
    trace_poly: Polynomial<FieldElement<F>>,
    primitive_root: FieldElement<F>,
}

impl<'a, A: AIR<F>, F: IsField> ConstraintEvaluator<'a, A, F> {
    pub fn new(
        air: &A,
        trace_poly: Polynomial<FieldElement<F>>,
        primitive_root: FieldElement<F>,
    ) -> Self {
        let boundary_constraints = air.compute_boundary_constraints();

        Self {
            air,
            boundary_constraints,
            trace_poly,
            primitive_root,
        }
    }

    pub fn evaluate(
        &self,
        lde_trace: TraceTable<F>,
        lde_domain: Vec<FieldElement<F>>,
    ) -> ConstraintEvaluationTable<F> {
        // Get all divisors in a vector
        // The first divisors appearing in the vector will be transition ones
        // and the last the one from the boundary constraints.
        // divisors.push(
        //     self.boundary_constraints
        //         .compute_zerofier(&self.primitive_root),
        // );

        // Initialize evaluation table
        let mut evaluation_table = ConstraintEvaluationTable::new(
            self.air.context().num_transition_constraints() + 1,
            &lde_domain,
        );

        let boundary_constraints = self.boundary_constraints;
        let domain = boundary_constraints.generate_roots_of_unity(&self.primitive_root);

        // Hard-coded for fibonacci -> trace has one column, hence col value is 0.
        let values = boundary_constraints.values(0);

        let max_degree =
            self.trace_poly.degree() * self.air.context().transition_degrees.iter().max().unwrap();

        let max_degree_power_of_two = next_power_of_two(max_degree);

        let boundary_poly = self.trace_poly - Polynomial::interpolate(&domain, &values);

        let alpha = FE::one();
        let beta = FE::one();

        let alpha_and_beta_coefficients = vec![(alpha, beta)];

        // Iterate over trace and domain and compute transitions
        for (i, d) in lde_domain.iter().enumerate() {
            let frame = Frame::read_from_trace(&lde_trace, i);

            let evaluations = self.air.compute_transition(&frame);
            let evaluations = self.compute_transition_evaluations(
                evaluations.as_slice(),
                alpha_and_beta_coefficients.as_slice(),
                max_degree_power_of_two,
                d,
            );

            let alpha = FE::one();
            let beta = FE::one();

            // Append evaluation for boundary constraints
            let boundary_evaluation = boundary_poly.evaluate(d)
                * (alpha * d.pow(max_degree - boundary_poly.degree() + beta));

            let boundary_zerofier = self
                .boundary_constraints
                .compute_zerofier(&self.primitive_root);

            evaluations.push(boundary_evaluation / boundary_zerofier.evaluate(d));

            evaluation_table.evaluations.push(evaluations);
        }

        evaluation_table
    }

    fn compute_transition_evaluations(
        &self,
        evaluations: &[FieldElement<F>],
        alpha_and_beta_coefficients: &[(FieldElement<F>, FieldElement<F>)],
        // composition_coeffs: (FieldElement<F>, FieldElement<F>),
        max_degree: u64,
        x: &FieldElement<F>,
    ) -> Vec<FieldElement<F>> {
        let transition_degrees = self.air.context().transition_degrees;

        // TODO: Fiat-Shamir
        // let alpha = FE::one();
        // let beta = FE::one();

        let divisors = self.air.transition_divisors();

        let mut evaluations = Vec::new();
        for (((eval, degree), div), (alpha, beta)) in evaluations
            .iter()
            .zip(transition_degrees)
            .zip(divisors)
            .zip(alpha_and_beta_coefficients)
        {
            let numerator = eval * (alpha * x.pow(max_degree - (degree as u64)) + beta);
            let result = numerator / div.evaluate(&x);
            evaluations.push(result);
        }

        evaluations
    }
}

fn next_power_of_two(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        (u64::MAX >> (n - 1).leading_zeros()) + 1
    }
}
