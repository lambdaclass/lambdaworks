use super::boundary::BoundaryConstraints;
use crate::air::AIR;
use std::marker::PhantomData;

use crate::{domain::Domain, frame::Frame, trace::LDETraceTable};
use itertools::Itertools;
use lambdaworks_math::circle::point::CirclePoint;
use lambdaworks_math::circle::polynomial::{evaluate_point, interpolate_cfft};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;

pub struct ConstraintEvaluator<A: AIR> {
    boundary_constraints: BoundaryConstraints,
    phantom: PhantomData<A>,
}

// See https://vitalik.eth.limo/general/2024/07/23/circlestarks.html (Section: Quetienting).
// https://github.com/ethereum/research/blob/master/circlestark/line_functions.py#L10
pub fn line(
    eval_point: &CirclePoint<Mersenne31Field>,
    first_vanish_point: &CirclePoint<Mersenne31Field>,
    second_vanish_point: &CirclePoint<Mersenne31Field>,
) -> FieldElement<Mersenne31Field> {
    (first_vanish_point.y - second_vanish_point.y) * eval_point.x
        + (second_vanish_point.x - first_vanish_point.x) * eval_point.y
        + (first_vanish_point.x * second_vanish_point.y
            - first_vanish_point.y * second_vanish_point.x)
}

// See https://vitalik.eth.limo/general/2024/07/23/circlestarks.html (Section: Quetienting).
// https://github.com/ethereum/research/blob/master/circlestark/line_functions.py#L16
// Evaluates the polybomial I at eval_point. I is the polynomial such that I(point_1) = value_1 and
// I(point_2) = value_2.
pub fn interpolant(
    point_1: &CirclePoint<Mersenne31Field>,
    point_2: &CirclePoint<Mersenne31Field>,
    value_1: FieldElement<Mersenne31Field>,
    value_2: FieldElement<Mersenne31Field>,
    eval_point: &CirclePoint<Mersenne31Field>,
) -> FieldElement<Mersenne31Field> {
    let dx = point_2.x - point_1.x;
    let dy = point_2.y - point_1.y;
    // CHECK: can dx^2 + dy^2 = 0 even if dx!=0 and dy!=0 ? (using that they are FE of Mersenne31).
    let invdist = (dx * dx + dy * dy).inv().unwrap();
    let dot = (eval_point.x - point_1.x) * dx + (eval_point.y - point_1.y) * dy;
    value_1 + (value_2 - value_1) * dot * invdist
}

impl<A: AIR> ConstraintEvaluator<A> {
    pub fn new(air: &A) -> Self {
        let boundary_constraints = air.boundary_constraints();

        Self {
            boundary_constraints,
            phantom: PhantomData,
        }
    }

    pub(crate) fn evaluate(
        &self,
        air: &A,
        lde_trace: &LDETraceTable,
        domain: &Domain,
        transition_coefficients: &[FieldElement<Mersenne31Field>],
        boundary_coefficients: &[FieldElement<Mersenne31Field>],
    ) -> Vec<FieldElement<Mersenne31Field>> {
        let boundary_constraints = &self.boundary_constraints;
        let number_of_b_constraints = boundary_constraints.constraints.len();

        let boundary_zerofiers_inverse_evaluations: Vec<Vec<FieldElement<Mersenne31Field>>> =
            boundary_constraints
                .constraints
                .chunks(2)
                .map(|chunk| {
                    let first_constraint = &chunk[0];
                    let second_constraint = &chunk[1];
                    let first_vanish_point = &domain.trace_coset_points[first_constraint.step];
                    let second_vanish_point = &domain.trace_coset_points[second_constraint.step];
                    let mut evals = domain
                        .lde_coset_points
                        .iter()
                        .map(|eval_point| line(eval_point, first_vanish_point, second_vanish_point))
                        .collect::<Vec<FieldElement<Mersenne31Field>>>();
                    FieldElement::inplace_batch_inverse(&mut evals).unwrap();
                    evals
                })
                .collect::<Vec<Vec<FieldElement<Mersenne31Field>>>>();

        let boundary_polys_evaluations: Vec<Vec<FieldElement<Mersenne31Field>>> =
            boundary_constraints
                .constraints
                .chunks(2)
                .map(|chunk| {
                    let first_constraint = &chunk[0];
                    let second_constraint = &chunk[1];
                    let first_vanish_point = &domain.trace_coset_points[first_constraint.step];
                    let first_value = first_constraint.value;
                    let second_vanish_point = &domain.trace_coset_points[second_constraint.step];
                    let second_value = second_constraint.value;
                    let mut evals = domain
                        .lde_coset_points
                        .iter()
                        .map(|eval_point| {
                            interpolant(
                                first_vanish_point,
                                second_vanish_point,
                                first_value,
                                second_value,
                                eval_point,
                            )
                        })
                        .collect::<Vec<FieldElement<Mersenne31Field>>>();
                    FieldElement::inplace_batch_inverse(&mut evals).unwrap();
                    evals
                })
                .collect::<Vec<Vec<FieldElement<Mersenne31Field>>>>();

        // let boundary_polys_evaluations = boundary_constraints
        //     .constraints
        //     .iter()
        //     .map(|constraint| {
        //         (0..lde_trace.num_rows())
        //             .map(|row| {
        //                 let v = lde_trace.table.get(row, constraint.col);
        //                 v - &constraint.value
        //             })
        //             .collect_vec()
        //     })
        //     .collect_vec();

        // ---------------  BEGIN TESTING ----------------------------
        // Interpolate lde trace evaluations.
        // let boundary_poly_coefficients = boundary_polys_evaluations
        //     .iter()
        //     .map(|evals| interpolate_cfft(evals.to_vec()))
        //     .collect::<Vec<Vec<FieldElement<Mersenne31Field>>>>();

        // Evaluate lde trace interpolating polynomial in trace domain.
        // for point in &domain.trace_coset_points {
        //     println!(
        //         "{:?}",
        //         evaluate_point(&boundary_poly_coefficients[0], &point)
        //     );
        // }
        // ---------------  END TESTING ----------------------------

        let boundary_eval_iter = 0..domain.lde_coset_points.len();

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

        // Iterate over all LDE domain and compute the part of the composition polynomial
        // related to the transition constraints and add it to the already computed part of the
        // boundary constraints.

        let zerofiers_evals = air.transition_zerofier_evaluations(domain);

        // ---------------  BEGIN TESTING ----------------------------
        // Interpolate lde trace evaluations.
        let zerofier_poly_coefficients = zerofiers_evals
            .iter()
            .map(|evals| interpolate_cfft(evals.to_vec()))
            .collect::<Vec<Vec<FieldElement<Mersenne31Field>>>>();

        // Evaluate lde trace interpolating polynomial in trace domain.
        // This should print all zeroes except in the end exceptions points.
        for point in &domain.trace_coset_points {
            println!(
                "{:?}",
                evaluate_point(&zerofier_poly_coefficients[0], &point)
            );
        }
        // ---------------  END TESTING ----------------------------

        let evaluations_t_iter = 0..domain.lde_coset_points.len();

        let evaluations_t = evaluations_t_iter
            .zip(boundary_evaluation)
            .map(|(i, boundary)| {
                let frame = Frame::read_from_lde(lde_trace, i, &air.context().transition_offsets);

                // Compute all the transition constraints at this point of the LDE domain.
                let evaluations_transition = air.compute_transition_prover(&frame);

                // Add each term of the transition constraints to the composition polynomial, including the zerofier,
                // the challenge and the exemption polynomial if it is necessary.
                let acc_transition = itertools::izip!(
                    evaluations_transition,
                    &zerofiers_evals,
                    transition_coefficients
                )
                .fold(FieldElement::zero(), |acc, (eval, zerof_eval, beta)| {
                    acc + &zerof_eval[i] * eval * beta
                });

                acc_transition + boundary
            })
            .collect();

        evaluations_t
    }
}
