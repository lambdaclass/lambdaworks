use crate::frame::Frame;
use crate::trace::TraceTable;

use super::domain::Domain;
use super::traits::AIR;
use lambdaworks_math::fft::polynomial::FFTPoly;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
use log::{error, info};

/// Validates that the trace is valid with respect to the supplied AIR constraints
pub fn validate_trace<F: IsFFTField, A: AIR<Field = F>>(
    air: &A,
    trace_polys: &[Polynomial<FieldElement<A::Field>>],
    domain: &Domain<A::Field>,
    rap_challenges: &A::RAPChallenges,
) -> bool {
    info!("Starting constraints validation over trace...");
    let mut ret = true;

    let trace_columns: Vec<_> = trace_polys
        .iter()
        .map(|poly| {
            poly.evaluate_fft(1, Some(domain.interpolation_domain_size))
                .unwrap()
        })
        .collect();
    let trace = TraceTable::new_from_cols(&trace_columns);

    // --------- VALIDATE BOUNDARY CONSTRAINTS ------------
    air.boundary_constraints(rap_challenges)
        .constraints
        .iter()
        .for_each(|constraint| {
            let col = constraint.col;
            let step = constraint.step;
            let boundary_value = constraint.value.clone();
            let trace_value = trace.get(step, col);

            if boundary_value != trace_value {
                ret = false;
                error!("Boundary constraint inconsistency - Expected value {} in step {} and column {}, found: {}", boundary_value.representative(), step, col, trace_value.representative());
            }
        });

    // --------- VALIDATE TRANSITION CONSTRAINTS -----------
    let n_transition_constraints = air.context().num_transition_constraints();
    let transition_exemptions = &air.context().transition_exemptions;

    let exemption_steps: Vec<usize> = vec![trace.n_rows(); n_transition_constraints]
        .iter()
        .zip(transition_exemptions)
        .map(|(trace_steps, exemptions)| trace_steps - exemptions)
        .collect();

    // Iterate over trace and compute transitions
    for step in 0..trace.n_rows() {
        let frame = Frame::read_from_trace(&trace, step, 1, &air.context().transition_offsets);

        let evaluations = air.compute_transition(&frame, rap_challenges);
        // Iterate over each transition evaluation. When the evaluated step is not from
        // the exemption steps corresponding to the transition, it should have zero as a
        // result
        evaluations.iter().enumerate().for_each(|(i, eval)| {
            // Check that all the transition constraint evaluations of the trace are zero.
            // We don't take into account the transition exemptions.
            if step < exemption_steps[i] && eval != &FieldElement::<F>::zero() {
                ret = false;
                error!(
                    "Inconsistent evaluation of transition {} in step {} - expected 0, got {}",
                    i,
                    step,
                    eval.representative()
                );
            }
        })
    }
    info!("Constraints validation check ended");
    ret
}

pub fn check_boundary_polys_divisibility<F: IsFFTField>(
    boundary_polys: Vec<Polynomial<FieldElement<F>>>,
    boundary_zerofiers: Vec<Polynomial<FieldElement<F>>>,
) {
    for (i, (poly, z)) in boundary_polys
        .iter()
        .zip(boundary_zerofiers.iter())
        .enumerate()
    {
        let (_, b) = poly.clone().long_division_with_remainder(z);
        if b != Polynomial::zero() {
            error!("Boundary poly {} is not divisible by its zerofier", i);
        }
    }
}
