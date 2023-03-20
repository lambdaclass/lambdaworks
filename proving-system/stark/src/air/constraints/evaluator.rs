use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

use crate::air::{frame::Frame, trace::TraceTable, AIR};

use super::{boundary::BoundaryConstraints, evaluation_table::ConstraintEvaluationTable, helpers};

pub struct ConstraintEvaluator<F: IsField, A: AIR> {
    air: A,
    boundary_constraints: BoundaryConstraints<F>,
    trace_poly: Polynomial<FieldElement<F>>,
    primitive_root: FieldElement<F>,
}

impl<F: IsField, A: AIR + AIR<Field = F>> ConstraintEvaluator<F, A> {
    pub fn new(
        air: &A,
        trace_poly: &Polynomial<FieldElement<F>>,
        primitive_root: &FieldElement<F>,
    ) -> Self {
        let boundary_constraints = air.compute_boundary_constraints();

        Self {
            air: air.clone(),
            boundary_constraints,
            trace_poly: trace_poly.clone(),
            primitive_root: primitive_root.clone(),
        }
    }

    // TODO: This does not work for multiple columns
    pub fn evaluate(
        &self,
        lde_trace: &TraceTable<F>,
        lde_domain: &[FieldElement<F>],
        alpha_and_beta_transition_coefficients: &[(FieldElement<F>, FieldElement<F>)],
        alpha_and_beta_boundary_coefficients: (&FieldElement<F>, &FieldElement<F>),
    ) -> ConstraintEvaluationTable<F> {
        let mut evaluation_table = ConstraintEvaluationTable::new(
            self.air.context().num_transition_constraints() + 1,
            lde_domain,
        );

        let boundary_constraints = &self.boundary_constraints;
        let domain = boundary_constraints.generate_roots_of_unity(&self.primitive_root);

        // TODO: Unhardcode this
        // Hard-coded for fibonacci -> trace has one column, hence col value is 0.
        let values = boundary_constraints.values(0);

        let max_degree = self.trace_poly.degree()
            * self
                .air
                .context()
                .transition_degrees()
                .iter()
                .max()
                .unwrap();

        let max_degree_power_of_two = helpers::next_power_of_two(max_degree as u64);

        let boundary_poly = &self.trace_poly - &Polynomial::interpolate(&domain, &values);

        let (boundary_alpha, boundary_beta) = alpha_and_beta_boundary_coefficients;

        let blowup_factor = self.air.blowup_factor();

        // Iterate over trace and domain and compute transitions
        for (i, d) in lde_domain.iter().enumerate() {
            let frame = Frame::read_from_trace(
                lde_trace,
                i,
                blowup_factor,
                &self.air.context().transition_offsets,
            );

            let mut evaluations = self.air.compute_transition(&frame);
            evaluations = Self::compute_transition_evaluations(
                &self.air,
                &evaluations,
                alpha_and_beta_transition_coefficients,
                max_degree_power_of_two,
                d,
            );

            // Append evaluation for boundary constraints
            let boundary_evaluation = boundary_poly.evaluate(d)
                * (boundary_alpha
                    * d.pow(max_degree_power_of_two - (boundary_poly.degree() as u64))
                    + boundary_beta);

            let boundary_zerofier = self
                .boundary_constraints
                .compute_zerofier(&self.primitive_root);

            evaluations.push(boundary_evaluation / boundary_zerofier.evaluate(d));

            evaluation_table.evaluations.push(evaluations);
        }

        evaluation_table
    }

    /// Given `evaluations` C_i(x) of the trace polynomial composed with the constraint
    /// polynomial at a certain point `x`, computes the following evaluations and returns them:
    ///
    /// C_i(x) (alpha_i * x^(D - D_i) + beta_i) / (Z_i(x))
    ///
    /// where Z is the zerofier of the `i`-th transition constraint polynomial.
    ///
    /// This method is called in two different scenarios. The first is when the prover
    /// is building these evaluations to compute the composition and DEEP composition polynomials.
    /// The second one is when the verifier needs to check the consistency between the trace and
    /// the composition polynomial. In that case the `evaluations` are over an *out of domain* frame
    /// (in the fibonacci example they are evaluations on the points `z`, `zg`, `zg^2`).
    pub fn compute_transition_evaluations(
        air: &A,
        evaluations: &[FieldElement<F>],
        alpha_and_beta_coefficients: &[(FieldElement<F>, FieldElement<F>)],
        max_degree: u64,
        x: &FieldElement<F>,
    ) -> Vec<FieldElement<F>> {
        let transition_degrees = air.context().transition_degrees();

        let divisors = air.transition_divisors();

        let mut ret = Vec::new();
        for (((eval, degree), div), (alpha, beta)) in evaluations
            .iter()
            .zip(transition_degrees)
            .zip(divisors)
            .zip(alpha_and_beta_coefficients)
        {
            let numerator = eval * (alpha * x.pow(max_degree - (degree as u64)) + beta);
            let result = numerator / div.evaluate(x);
            ret.push(result);
        }

        ret
    }
}
