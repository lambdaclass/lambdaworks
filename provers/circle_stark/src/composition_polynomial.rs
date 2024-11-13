use crate::air::AIR;
use crate::{
    constraints::boundary::BoundaryConstraints, domain::Domain, frame::Frame, trace::LDETraceTable,
};
use itertools::Itertools;
use lambdaworks_math::{
    circle::{
        point::CirclePoint,
        polynomial::{evaluate_cfft, evaluate_point, interpolate_cfft},
    },
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field},
};

pub(crate) fn evaluate_cp<A: AIR>(
    air: &A,
    lde_trace: &LDETraceTable,
    domain: &Domain,
    transition_coefficients: &[FieldElement<Mersenne31Field>],
    boundary_coefficients: &[FieldElement<Mersenne31Field>],
) -> Vec<FieldElement<Mersenne31Field>> {
    // >>> First, we compute the part of the Composition Polynomial related to the boundary constraints.

    let boundary_constraints = &air.boundary_constraints();
    let number_of_b_constraints = boundary_constraints.constraints.len();
    let trace_coset = &domain.trace_coset_points;
    let lde_coset = &domain.lde_coset_points;

    // For each pair of boundary constraints, we calculate the denominator's evaluations.
    let boundary_zerofiers_inverse_evaluations =
        boundary_constraints.evaluate_zerofiers(&trace_coset, &lde_coset);

    // For each pair of boundary constraints, we calculate the numerator's evaluations.
    let boundary_polys_evaluations =
        boundary_constraints.evaluate_poly_constraints(&trace_coset, &lde_coset, lde_trace);

    // We begin to construct the cp by adding each numerator mulpitlied by the denominator and the beta coefficient.
    let cp_boundary: Vec<FieldElement<Mersenne31Field>> = (0..lde_coset.len())
        .map(|domain_index| {
            (0..number_of_b_constraints)
                .step_by(2)
                .zip(boundary_coefficients)
                .fold(FieldElement::zero(), |acc, (constraint_index, beta)| {
                    acc + &boundary_zerofiers_inverse_evaluations[constraint_index][domain_index]
                        * beta
                        * &boundary_polys_evaluations[constraint_index][domain_index]
                })
        })
        .collect();

    // >>> Now we compute the part of the CP related to the transition constraints and add it to the already
    // computed part of the boundary constraints.

    // For each transition constraint, we calulate its zerofier's evaluations.
    let transition_zerofiers_inverse_evaluations = air.transition_zerofier_evaluations(domain);

    //
    let cp_evaluations = (0..lde_coset.len())
        .zip(cp_boundary)
        .map(|(eval_index, boundary_eval)| {
            let frame =
                Frame::read_from_lde(lde_trace, eval_index, &air.context().transition_offsets);
            let transition_poly_evaluations = air.compute_transition_prover(&frame);
            let transition_polys_accumulator = itertools::izip!(
                transition_poly_evaluations,
                &transition_zerofiers_inverse_evaluations,
                transition_coefficients
            )
            .fold(
                FieldElement::zero(),
                |acc, (transition_eval, zerof_eval, beta)| {
                    acc + &zerof_eval[eval_index] * transition_eval * beta
                },
            );
            transition_polys_accumulator + boundary_eval
        })
        .collect();

    cp_evaluations
}

#[cfg(test)]
mod test {

    use crate::examples::simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs};

    use super::*;

    type FE = FieldElement<Mersenne31Field>;

    fn build_fibonacci_example() {}

    #[test]
    fn boundary_zerofiers_vanish_correctly() {
        // Build Fibonacci Example
        let trace = simple_fibonacci::fibonacci_trace([FE::one(), FE::one()], 32);
        let pub_inputs = FibonacciPublicInputs {
            a0: FE::one(),
            a1: FE::one(),
        };
        let air = FibonacciAIR::new(trace.n_rows(), &pub_inputs);
        let boundary_constraints = air.boundary_constraints();
        let domain = Domain::new(&air);

        // Calculate the boundary zerofiers evaluations (L function).
        let boundary_zerofiers_inverse_evaluations = boundary_constraints
            .evaluate_zerofiers(&domain.trace_coset_points, &domain.lde_coset_points);

        // Interpolate the boundary zerofiers evaluations.
        let boundary_zerofiers_coeff = boundary_zerofiers_inverse_evaluations
            .iter()
            .map(|evals| {
                let mut inverse_evals = evals.clone();
                FieldElement::inplace_batch_inverse(&mut inverse_evals).unwrap();
                interpolate_cfft(inverse_evals.to_vec())
            })
            .collect::<Vec<Vec<FieldElement<Mersenne31Field>>>>();

        // Since simple fibonacci only has one pair of boundary constraints we only check that
        // the corresponding polynomial evaluates 0 in the first two coset points and different from 0
        // in the rest of the points.
        assert_eq!(
            evaluate_point(&boundary_zerofiers_coeff[0], &domain.trace_coset_points[0]),
            FE::zero()
        );

        assert_eq!(
            evaluate_point(&boundary_zerofiers_coeff[0], &domain.trace_coset_points[1]),
            FE::zero()
        );

        for point in domain.trace_coset_points.iter().skip(2) {
            assert_ne!(
                evaluate_point(&boundary_zerofiers_coeff[0], &point),
                FE::zero()
            );
        }
    }

    #[test]
    fn boundary_polys_vanish_correctly() {
        // Build Fibonacci Example
        let trace = simple_fibonacci::fibonacci_trace([FE::one(), FE::one()], 32);
        let pub_inputs = FibonacciPublicInputs {
            a0: FE::one(),
            a1: FE::one(),
        };
        let air = FibonacciAIR::new(trace.n_rows(), &pub_inputs);
        let boundary_constraints = air.boundary_constraints();
        let domain = Domain::new(&air);

        // Evaluate each polynomial in the lde domain.
        let lde_trace = LDETraceTable::new(trace.table.data.clone(), 1, 1);

        // Calculate boundary polynomials evaluations (the polynomial f - I).
        let boundary_polys_evaluations = boundary_constraints.evaluate_poly_constraints(
            &domain.trace_coset_points,
            &domain.lde_coset_points,
            &lde_trace,
        );

        // Interpolate the boundary polynomials evaluations.
        let boundary_poly_coeff = boundary_polys_evaluations
            .iter()
            .map(|evals| interpolate_cfft(evals.to_vec()))
            .collect::<Vec<Vec<FieldElement<Mersenne31Field>>>>();

        // Since simple fibonacci only has one pair of boundary constraints we only check that
        // the corresponding polynomial evaluates 0 in the first two coset points and different from 0
        // in the rest of the points.
        assert_eq!(
            evaluate_point(&boundary_poly_coeff[0], &domain.trace_coset_points[0]),
            FE::zero()
        );

        assert_eq!(
            evaluate_point(&boundary_poly_coeff[0], &domain.trace_coset_points[1]),
            FE::zero()
        );

        for point in domain.trace_coset_points.iter().skip(2) {
            assert_ne!(evaluate_point(&boundary_poly_coeff[0], &point), FE::zero());
        }
    }
}
