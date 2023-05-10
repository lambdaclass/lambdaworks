use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    helpers,
    polynomial::Polynomial,
};

use crate::air::{frame::Frame, trace::TraceTable, AIR};
use std::iter::zip;

use super::evaluation_table::ConstraintEvaluationTable;

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

    pub fn evaluate(
        &self,
        lde_trace: &TraceTable<F>,
        lde_domain: &[FieldElement<F>],
        transition_coefficients: &[(FieldElement<F>, FieldElement<F>)],
        boundary_coefficients: &[(FieldElement<F>, FieldElement<F>)],
        aux_segments_rand_elements: Option<&Vec<Vec<FieldElement<F>>>>,
    ) -> ConstraintEvaluationTable<F> {
        // The + 1 is for the boundary constraints column
        let mut evaluation_table = ConstraintEvaluationTable::new(
            self.air.context().num_transition_constraints() + 1,
            lde_domain,
        );

        let n_trace_colums = self.main_trace_polys.len();
        let main_boundary_constraints = self.air.boundary_constraints();

        // Main trace boundary polys
        let main_boundary_domains =
            main_boundary_constraints.generate_roots_of_unity(&self.primitive_root, n_trace_colums);
        let main_boundary_values = main_boundary_constraints.values(n_trace_colums);
        let mut boundary_polys: Vec<Polynomial<FieldElement<F>>> =
            zip(main_boundary_domains, main_boundary_values)
                .zip(self.main_trace_polys)
                .map(|((xs, ys), trace_poly)| {
                    if xs.is_empty() {
                        trace_poly.clone()
                    } else {
                        trace_poly - &Polynomial::interpolate(&xs, &ys)
                    }
                })
                .collect();
        let mut boundary_zerofiers: Vec<Polynomial<FieldElement<F>>> = (0..n_trace_colums)
            .map(|col| main_boundary_constraints.compute_zerofier(&self.primitive_root, col))
            .collect();
        // Get the max degree of boundary polys of main segments
        let mut bound_polys_max_degree =
            boundary_polys_max_degree(&boundary_polys, &boundary_zerofiers);

        // Initialize trace_polys vector with only the main trace polynomials.
        let mut trace_polys = self.main_trace_polys.to_vec();

        // Get main transition degrees. When there are auxiliary transition constraints,
        // this vector is updated with their corresponding degrees later on.
        let mut transition_degrees = self.air.context().transition_degrees();

        // Auxiliary trace boundary polys
        let n_aux_segments = self.air.num_aux_segments();
        // let mut aux_rand_elements = Vec::with_capacity(n_aux_segments);
        if self.air.is_multi_segment() {
            let Some(aux_transition_degrees) = self.air.context().aux_transition_degrees
                else {
                    panic!("AIR context inconsistency - AIR is multi-segment but there are no auxiliary transition degrees set.");
                };

            // Update trace_polys vector with auxiliary trace polynomials.
            self.aux_trace_polys
                .iter()
                .for_each(|aux_poly| trace_polys.extend_from_slice(aux_poly));

            transition_degrees.extend_from_slice(&aux_transition_degrees);

            let mut aux_boundary_polys = Vec::with_capacity(n_aux_segments);
            let mut aux_boundary_zerofiers = Vec::with_capacity(n_aux_segments);
            (0..n_aux_segments).for_each(|segment_idx| {
                // Sample random elements needed for construction of the aux segment.
                // let aux_segment_rand_elements =
                //     self.air.aux_segment_rand_coeffs(segment_idx, transcript);
                let aux_segment_rand_elements = &aux_segments_rand_elements.unwrap()[segment_idx];

                // Get interpolated polynomials for the aux segment. There will be one polynomial
                // for each column of the aux segment.
                let aux_segment_polys = &self.aux_trace_polys[segment_idx];

                // Get aux segment boundary constraint. The random elements are needed for setting
                // them up as a part of the RAP.
                let aux_segment_boundary_constraints = self
                    .air
                    .aux_boundary_constraints(segment_idx, aux_segment_rand_elements);

                // Get the domains where the boundary constraints are to be applied and their
                // corresponding values.
                let aux_segment_width = self.air.aux_segment_width(segment_idx);
                let aux_segment_boundary_domains = aux_segment_boundary_constraints
                    .generate_roots_of_unity(&self.primitive_root, aux_segment_width);
                let aux_segment_boundary_values =
                    aux_segment_boundary_constraints.values(aux_segment_width);

                // Interpolate to find the numerator of the boundary quotient:
                //      Ni_aux(x) = ti_aux(x) - Bi_aux(x)
                // where,
                //    * ti_aux(x): Auxiliary trace polynomial corresponding to column `i` of the segment.
                //    * Bi_aux(x): Polynomial obtained from interpolating aux boundary constraints in the column `i` of the segment.
                let aux_segment_boundary_polys: Vec<Polynomial<FieldElement<F>>> =
                    zip(aux_segment_boundary_domains, aux_segment_boundary_values)
                        .zip(aux_segment_polys)
                        .map(|((xs, ys), aux_trace_poly)| {
                            aux_trace_poly - &Polynomial::interpolate(&xs, &ys)
                        })
                        .collect();

                // Compute zerofiers for every `Ni_aux(x)`
                let aux_segment_boundary_zerofiers: Vec<Polynomial<FieldElement<F>>> = (0
                    ..aux_segment_width)
                    .map(|aux_col| {
                        aux_segment_boundary_constraints
                            .compute_zerofier(&self.primitive_root, aux_col)
                    })
                    .collect();

                // aux_rand_elements.push(aux_segment_rand_elements);
                aux_boundary_polys.push(aux_segment_boundary_polys);
                aux_boundary_zerofiers.push(aux_segment_boundary_zerofiers);
            });

            // NOTE: This is not very efficient, we may want to do it another way.
            // The polynomials vectors are flattened so that it is easier to operate with them.
            let flattened_aux_boundary_polys: Vec<Polynomial<FieldElement<F>>> =
                aux_boundary_polys.into_iter().flatten().collect();
            let flattened_aux_boundary_zerofier: Vec<Polynomial<FieldElement<F>>> =
                aux_boundary_zerofiers.into_iter().flatten().collect();
            let aux_boundary_polys_max_degree = boundary_polys_max_degree(
                &flattened_aux_boundary_polys,
                &flattened_aux_boundary_zerofier,
            );

            boundary_polys.extend_from_slice(&flattened_aux_boundary_polys);
            boundary_zerofiers.extend_from_slice(&flattened_aux_boundary_zerofier);
            bound_polys_max_degree =
                std::cmp::max(bound_polys_max_degree, aux_boundary_polys_max_degree);
        }

        let transition_zerofiers = self.air.transition_divisors();
        let transition_polys_max_degree =
            transition_polys_max_degree(&transition_degrees, &trace_polys, &transition_zerofiers);
        let max_degree = std::cmp::max(transition_polys_max_degree, bound_polys_max_degree);
        let max_degree_power_of_two = helpers::next_power_of_two(max_degree as u64);

        let blowup_factor = self.air.blowup_factor();
        // Iterate over trace and domain and compute transitions
        for (step, d) in lde_domain.iter().enumerate() {
            let main_frame = Frame::read_from_trace(&self.air, lde_trace, blowup_factor, step);

            let mut evaluations = self.air.compute_transition(&main_frame);

            // Compute auxiliary transitions if needed, extending the evaluations vector with them.
            if self.air.is_multi_segment() {
                (0..n_aux_segments).for_each(|segment_idx| {
                    let aux_frame =
                        Frame::read_from_aux_segment(&self.air, lde_trace, step, segment_idx);
                    let aux_evaluations = self.air.compute_aux_transition(
                        &main_frame,
                        &aux_frame,
                        &aux_segments_rand_elements.unwrap()[segment_idx],
                    );
                    evaluations.extend_from_slice(&aux_evaluations);
                });
            }

            evaluations = Self::compute_constraint_composition_poly_evaluations(
                &self.air,
                &evaluations,
                transition_coefficients,
                max_degree_power_of_two,
                d,
            );

            // Merge all boundary terms evaluations into a single one.
            let boundary_evaluation = zip(&boundary_polys, &boundary_zerofiers)
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

// -------- HELPER FUNCTIONS ---------------

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
