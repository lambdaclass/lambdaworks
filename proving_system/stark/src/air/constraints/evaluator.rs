use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    helpers,
    polynomial::Polynomial,
};

use crate::air::{frame::Frame, trace::TraceTable, AIR};
use std::iter::zip;

use super::{
    boundary::{BoundaryConstraint, BoundaryConstraints},
    evaluation_table::ConstraintEvaluationTable,
};

pub struct ConstraintEvaluator<'poly, F: IsFFTField, A: AIR> {
    air: A,
    main_trace_polys: &'poly [Polynomial<FieldElement<F>>],
    aux_trace_polys: &'poly [Vec<Polynomial<FieldElement<F>>>],
    primitive_root: FieldElement<F>,
}

impl<'poly, F: IsFFTField, A: AIR<Field = F>> ConstraintEvaluator<'poly, F, A> {
    pub fn new(
        air: &A,
        main_trace_polys: &'poly [Polynomial<FieldElement<F>>],
        aux_trace_polys: &'poly [Vec<Polynomial<FieldElement<F>>>],
        primitive_root: &FieldElement<F>,
    ) -> Self {
        Self {
            air: air.clone(),
            main_trace_polys,
            aux_trace_polys,
            primitive_root: primitive_root.clone(),
        }
    }

    pub fn evaluate<T: Transcript>(
        &self,
        lde_trace: &TraceTable<F>,
        lde_domain: &[FieldElement<F>],
        transition_coefficients: &[(FieldElement<F>, FieldElement<F>)],
        boundary_coefficients: &[(FieldElement<F>, FieldElement<F>)],
        transcript: &T,
    ) -> ConstraintEvaluationTable<F> {
        // The + 1 is for the boundary constraints column
        let mut evaluation_table = ConstraintEvaluationTable::new(
            self.air.context().num_transition_constraints() + 1,
            lde_domain,
        );

        let n_trace_colums = self.main_trace_polys.len();
        let main_boundary_constraints = self.air.boundary_constraints();
        // let aux_boundary_constraints = self.air.aux_boundary_constraints().to_vec();

        // let boundary_constraints_vec: Vec<BoundaryConstraint<F>> =
        //     vec![main_boundary_constraints, aux_boundary_constraints]
        //         .iter()
        //         .flatten()
        //         .collect();

        // let boundary_constraints = &self.boundary_constraints;

        let domains =
            main_boundary_constraints.generate_roots_of_unity(&self.primitive_root, n_trace_colums);
        let values = main_boundary_constraints.values(n_trace_colums);

        // Main trace boundary polys
        let main_boundary_polys: Vec<Polynomial<FieldElement<F>>> = zip(domains, values)
            .zip(self.main_trace_polys)
            .map(|((xs, ys), trace_poly)| trace_poly - &Polynomial::interpolate(&xs, &ys))
            .collect();
        let main_boundary_zerofiers: Vec<Polynomial<FieldElement<F>>> = (0..n_trace_colums)
            .map(|col| main_boundary_constraints.compute_zerofier(&self.primitive_root, col))
            .collect();

        // Auxiliary trace boundary polys
        let n_aux_segments = self.air.num_aux_segments();
        let mut aux_rand_elements = Vec::with_capacity(n_aux_segments);
        if self.air.is_multi_segment() {
            (0..n_aux_segments).for_each(|segment_idx| {
                let aux_segment_rand_elements =
                    self.air.aux_segment_rand_coeffs(segment_idx, transcript);
                let aux_segment_polys = &self.aux_trace_polys[segment_idx];
                let aux_segment_boundary_constraints = self
                    .air
                    .aux_boundary_constraints(segment_idx, &aux_segment_rand_elements);

                let aux_segment_width = self.air.aux_segment_width(segment_idx);
                let aux_segment_boundary_domains = aux_segment_boundary_constraints
                    .generate_roots_of_unity(&self.primitive_root, aux_segment_width);
                let aux_segment_boundary_values =
                    main_boundary_constraints.values(aux_segment_width);

                let aux_trace_polys = &self.aux_trace_polys[segment_idx];

                let aux_segment_boundary_polys: Vec<Polynomial<FieldElement<F>>> =
                    zip(aux_segment_boundary_domains, aux_segment_boundary_values)
                        .zip(aux_trace_polys)
                        .map(|((xs, ys), aux_trace_poly)| {
                            aux_trace_poly - &Polynomial::interpolate(&xs, &ys)
                        })
                        .collect();

                let aux_boundary_zerofiers: Vec<Polynomial<FieldElement<F>>> = (0
                    ..aux_segment_width)
                    .map(|aux_col| {
                        aux_segment_boundary_constraints
                            .compute_zerofier(&self.primitive_root, aux_col)
                    })
                    .collect();

                aux_rand_elements.push(aux_segment_rand_elements);
            })
        }

        let main_boundary_polys_max_degree =
            boundary_polys_max_degree(&main_boundary_polys, &main_boundary_zerofiers);

        let main_transition_degrees = self.air.context().transition_degrees();
        let transition_zerofiers = self.air.transition_divisors();
        let transition_polys_max_degree = transition_polys_max_degree(
            &main_transition_degrees,
            &self.main_trace_polys,
            &transition_zerofiers,
        );

        let max_degree = std::cmp::max(transition_polys_max_degree, main_boundary_polys_max_degree);

        let max_degree_power_of_two = helpers::next_power_of_two(max_degree as u64);

        let blowup_factor = self.air.blowup_factor();
        // Iterate over trace and domain and compute transitions
        for (i, d) in lde_domain.iter().enumerate() {
            let main_frame = Frame::read_from_trace(
                lde_trace,
                i,
                blowup_factor,
                &self.air.context().transition_offsets,
            );

            let mut evaluations = self.air.compute_transition(&main_frame);

            // Compute auxiliary transitions if needed
            if self.air.is_multi_segment() {
                let n_aux_segments = self.air.num_aux_segments();
                let aux_transition_offsets = self.air.context().aux_transition_offsets;
                let aux_transition_offsets = aux_transition_offsets.as_ref().unwrap();
                (0..n_aux_segments).for_each(|segment_idx| {
                    let aux_rand_elements =
                        self.air.aux_segment_rand_coeffs(segment_idx, transcript);
                    let aux_frame = Frame::read_from_aux_segment(
                        lde_trace,
                        i,
                        blowup_factor,
                        aux_transition_offsets,
                        segment_idx,
                    );
                    let aux_evaluations = self.air.compute_aux_transition(
                        &main_frame,
                        &aux_frame,
                        &aux_rand_elements,
                    );
                    evaluations.extend_from_slice(&aux_evaluations);
                });
            }

            // let aux_evaluations = self.air.compute_aux_transition(&aux_frame);
            evaluations = Self::compute_constraint_composition_poly_evaluations(
                &self.air,
                &evaluations,
                transition_coefficients,
                max_degree_power_of_two,
                d,
            );

            let boundary_evaluation = zip(&main_boundary_polys, &main_boundary_zerofiers)
                .enumerate()
                .map(|(index, (boundary_poly, boundary_zerofier))| {
                    let quotient_degree = boundary_poly.degree() - boundary_zerofier.degree();

                    let (boundary_alpha, boundary_beta) = boundary_coefficients[index].clone();

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
        constraint_coeffs: &[(FieldElement<F>, FieldElement<F>)],
        max_degree: u64,
        x: &FieldElement<F>,
    ) -> Vec<FieldElement<F>> {
        // TODO: We should get the trace degree in a better way because in some special cases
        // the trace degree may not be exactly the trace length - 1 but a smaller number.
        let trace_degree = air.trace_length() - 1;
        let transition_degrees = air.context().transition_degrees();
        let divisors = air.transition_divisors();

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

// Maybe this could be in a helpers or utils file
fn boundary_polys_max_degree<F>(
    boundary_polys: &[Polynomial<FieldElement<F>>],
    boundary_zerofier_polys: &[Polynomial<FieldElement<F>>],
) -> usize
where
    F: IsFFTField,
{
    boundary_polys
        .iter()
        .zip(boundary_zerofier_polys)
        .map(|(poly, zerofier)| poly.degree() - zerofier.degree())
        .max()
        .unwrap()
}

fn transition_polys_max_degree<F>(
    transition_degrees: &[usize],
    trace_polys: &[Polynomial<FieldElement<F>>],
    transition_zerofiers: &[Polynomial<FieldElement<F>>],
) -> usize
where
    F: IsFFTField,
{
    trace_polys
        .iter()
        .zip(transition_zerofiers)
        .zip(transition_degrees)
        .map(|((poly, zerofier), degree)| {
            (poly.degree() * degree).saturating_sub(zerofier.degree())
        })
        .max()
        .unwrap()
}
