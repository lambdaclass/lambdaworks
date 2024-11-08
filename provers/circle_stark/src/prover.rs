use crate::{air::AIR, config::FriMerkleTree, constraints::evaluator::ConstraintEvaluator, domain::Domain, frame::Frame, trace::{LDETraceTable, TraceTable}};
use std::marker::PhantomData;

use super::config::Commitment;
use lambdaworks_math::{
    circle::polynomial::{evaluate_cfft, evaluate_point, interpolate_cfft},
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field},
};

/// A default STARK prover implementing `IsStarkProver`.
pub struct Prover<A: AIR> {
    phantom: PhantomData<A>,
}

impl<A: AIR> IsStarkProver<A> for Prover<A> {}

#[derive(Debug)]
pub enum ProvingError {
    WrongParameter(String),
}

pub trait IsStarkProver<A: AIR> {
    fn prove(trace: &TraceTable, pub_inputs: &A::PublicInputs) {
        let air = A::new(trace.n_rows(), pub_inputs);
        let domain = Domain::new(&air);
        let lde_domain_length = domain.blowup_factor * domain.trace_length;

        // For each column, calculate the coefficients of the trace interpolating polynomial.
        let mut trace_polys = trace.compute_trace_polys();

        // Evaluate each polynomial in the lde domain.
        let lde_trace_evaluations = trace_polys
            .iter_mut()
            .map(|poly| {
                // Padding with zeros the coefficients of the polynomial, so we can evaluate it in the lde domain.
                poly.resize(lde_domain_length, FieldElement::zero());
                evaluate_cfft(poly.to_vec())
            })
            .collect::<Vec<Vec<FieldElement<Mersenne31Field>>>>();

        // TODO: Commit on lde trace evaluations.

        // --------- VALIDATE LDE TRACE EVALUATIONS ------------
        
        // Interpolate lde trace evaluations.
        let lde_coefficients = lde_trace_evaluations
            .iter()
            .map(|evals| interpolate_cfft(evals.to_vec()))
            .collect::<Vec<Vec<FieldElement<Mersenne31Field>>>>();

        // Evaluate lde trace interpolating polynomial in trace domain.
        for point in &domain.trace_coset_points {
            // println!("{:?}", evaluate_point(&lde_coefficients[0], &point));
        }

        // Crate a LDE_TRACE with a blow up factor of one, so its the same values as the trace.
        let not_lde_trace = LDETraceTable::new(trace.table.data.clone(), 1, 1);

        // --------- VALIDATE BOUNDARY CONSTRAINTS ------------
        air.boundary_constraints()
            .constraints
            .iter()
            .for_each(|constraint| {
                let col = constraint.col;
                let step = constraint.step;
                let boundary_value = constraint.value.clone();

                let trace_value = trace.table.get(step, col).clone();
            
            if boundary_value.clone() != trace_value {
                // println!("Boundary constraint inconsistency - Expected value {:?} in step {} and column {}, found: {:?}", boundary_value, step, col, trace_value);
            } else {
                // println!("Consistent Boundary constraint - Expected value {:?} in step {} and column {}, found: {:?}", boundary_value, step, col, trace_value)
            }
        });

        // --------- VALIDATE TRANSITION CONSTRAINTS -----------
        for row_index in 0..not_lde_trace.table.height - 2 {
            let frame = Frame::read_from_lde(&not_lde_trace, row_index, &air.context().transition_offsets);
            let evaluations = air.compute_transition_prover(&frame);
            // println!("Transition constraints evaluations: {:?}", evaluations);
        }


        let transition_coefficients: Vec<FieldElement<Mersenne31Field>> = vec![FieldElement::<Mersenne31Field>::one(); air.context().num_transition_constraints()];
        let boundary_coefficients: Vec<FieldElement<Mersenne31Field>> = vec![FieldElement::<Mersenne31Field>::one(); air.boundary_constraints().constraints.len()];
        
        // Compute the evaluations of the composition polynomial on the LDE domain.
        let lde_trace = LDETraceTable::from_columns(lde_trace_evaluations, domain.blowup_factor);
        let evaluator = ConstraintEvaluator::new(&air);
        let constraint_evaluations = evaluator.evaluate(
            &air,
            &lde_trace,
            &domain,
            &transition_coefficients,
            &boundary_coefficients,
         );
    }
}

// const BLOW_UP_FACTOR: usize = 2;

// pub fn prove(trace: Vec<FieldElement<Mersenne31Field>>) -> Commitment {

//     let lde_domain_size = trace.len() * BLOW_UP_FACTOR;

//     // Returns the coef of the interpolating polinomial of the trace on a natural domain.
//     let mut trace_poly = interpolate_cfft(trace);

//     // Padding with zeros the coefficients of the polynomial, so we can evaluate it in the lde domain.
//     trace_poly.resize(lde_domain_size, FieldElement::zero());
//     let lde_eval = evaluate_cfft(trace_poly);

//     let tree = FriMerkleTree::<Mersenne31Field>::build(&lde_eval).unwrap();
//     let commitment = tree.root;

//     commitment
// }

#[cfg(test)]
mod tests {

    use super::*;

    type FE = FieldElement<Mersenne31Field>;

    // #[test]
    // fn basic_test() {
    //     let trace = vec![
    //         FE::from(1),
    //         FE::from(2),
    //         FE::from(3),
    //         FE::from(4),
    //         FE::from(5),
    //         FE::from(6),
    //         FE::from(7),
    //         FE::from(8),
    //     ];

    //     let commitmet = prove(trace);
    //     println!("{:?}", commitmet);
    // }
}
