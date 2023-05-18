use std::iter::zip;

use lambdaworks_fft::polynomial::FFTPoly;
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
use log::{error, info};

use crate::{
    air::{frame::Frame, trace::TraceTable},
    Domain,
};

use super::AIR;

/// Validates that the trace is valid with respect to the supplied AIR constraints
pub fn validate_trace<F: IsFFTField, A: AIR<Field = F>>(
    air: &A,
    trace_polys: &[Polynomial<FieldElement<A::Field>>],
    domain: &Domain<A::Field>,
    public_input: &A::PublicInput,
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
    air.boundary_constraints(rap_challenges, public_input)
        .constraints
        .iter()
        .for_each(|constraint| {
            let col = constraint.col;
            let step = constraint.step;
            let boundary_value = constraint.value.clone();
            let trace_value = trace.get(step, col);

            if boundary_value != trace_value {
                ret = false;
                error!("Boundary constraint inconsistency - Expected value {:?} in step {} and column {}, found: {:?}", boundary_value, step, col, trace_value);
            }
        });

    // --------- VALIDATE TRANSITION CONSTRAINTS -----------
    let n_transition_constraints = air.context().num_transition_constraints();
    let transition_exemptions = air.context().transition_exemptions;

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
            if step < exemption_steps[i] && eval != &FieldElement::<F>::zero() {
                ret = false;
                error!(
                    "Inconsistent evaluation of transition {} in step {} - expected 0, got {:?}",
                    i, step, eval
                );
            }
        })
    }
    info!("Constraints validation check ended");
    ret
}

pub fn check_polynomial_divisibility<A>(
    air: &A,
    transition_evaluations: &[Vec<FieldElement<A::Field>>],
    transition_zerofiers: &[Polynomial<FieldElement<A::Field>>],
) where
    A: AIR,
    FieldElement<A::Field>: ByteConversion,
{
    let mut polys = Vec::new();
    let domain = Domain::new(air);
    for col_idx in 0..transition_evaluations[0].len() {
        let mut poly_evals = Vec::new();
        for transition_evaluation in transition_evaluations {
            poly_evals.push(transition_evaluation[col_idx].clone());
        }
        let poly = Polynomial::interpolate_offset_fft(&poly_evals, &domain.coset_offset).unwrap();
        polys.push(poly);
    }

    debug_assert_eq!(polys.len(), transition_zerofiers.len());

    zip(polys, transition_zerofiers)
        .enumerate()
        .for_each(|(i, (p, z))| {
            let (_, rem) = p.long_division_with_remainder(z);
            if rem != Polynomial::zero() {
                error!(
                    "Numerator of constraint poly for constraint {i} is not divisible by zerofier."
                );
            }
        });
}
