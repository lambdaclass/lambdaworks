use lambdaworks_math::polynomial::Polynomial;

use crate::{common::*, scs::SquareConstraintSystem};

#[derive(Debug)]
pub struct SquareSpamProgram {
    pub num_of_public_inputs: usize,
    pub num_of_gates: usize,
    pub u_poly: Vec<Polynomial<FrElement>>,
}

impl SquareSpamProgram {
    pub fn from_scs(scs: SquareConstraintSystem) -> Self {
        let num_of_gates = scs.number_of_constraints().next_power_of_two();

        let mut u_poly: Vec<Polynomial<FrElement>> = vec![];

        for var_idx in 0..scs.witness_size() {
            let poly = get_var_poly_from_scs(&scs, var_idx, num_of_gates);

            u_poly.push(poly);
        }

        Self {
            num_of_public_inputs: scs.number_of_inputs,
            num_of_gates,
            u_poly,
        }
    }
}

#[inline]
fn get_var_poly_from_scs(
    scs: &SquareConstraintSystem,
    var_idx: usize,
    num_of_gates: usize,
) -> Polynomial<FrElement> {
    let mut var_u = vec![FrElement::zero(); num_of_gates];

    for (constraint_idx, constraint) in scs.constraints.iter().enumerate() {
        var_u[constraint_idx] = constraint.u[var_idx].clone();
    }

    Polynomial::interpolate_fft::<FrField>(&var_u).unwrap()
}
