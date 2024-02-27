use lambdaworks_math::polynomial::Polynomial;

use crate::{common::*, scs::SquareConstraintSystem};

#[derive(Debug)]
pub struct SquareSpanProgram {
    pub num_of_public_inputs: usize,
    pub num_of_gates: usize,
    pub u_poly: Vec<Polynomial<FrElement>>,
}

impl SquareSpanProgram {
    pub fn calculate_h_coefficients(&self, w: &[FrElement]) -> Vec<FrElement> {
        let offset = &ORDER_R_MINUS_1_ROOT_UNITY;
        let degree = self.num_of_gates * 2;

        let u_eval = self.scale_and_accumulate_variable_polynomials(w, degree, offset);

        let t_poly =
            Polynomial::new_monomial(FrElement::one(), self.num_of_gates) - FrElement::one();
        let mut t = Polynomial::evaluate_offset_fft(&t_poly, 1, Some(degree), offset).unwrap();
        FrElement::inplace_batch_inverse(&mut t).unwrap();

        let h_evaluated = u_eval
            .iter()
            .zip(&t)
            .map(|(u, t)| (u * u - FrElement::one()) * t)
            .collect::<Vec<_>>();

        Polynomial::interpolate_offset_fft(&h_evaluated, offset)
            .unwrap()
            .coefficients()
            .to_vec()
    }

    // Compute U.w by summing up polynomials U[0].w_0, U[1].w_1, ..., U[n].w_n
    fn scale_and_accumulate_variable_polynomials(
        &self,
        w: &[FrElement],
        degree: usize,
        offset: &FrElement,
    ) -> Vec<FrElement> {
        Polynomial::evaluate_offset_fft(
            &(self
                .u_poly
                .iter()
                .zip(w)
                .map(|(poly, coeff)| poly.mul_with_ref(&Polynomial::new_monomial(coeff.clone(), 0)))
                .reduce(|poly1, poly2| poly1 + poly2)
                .unwrap()),
            1,
            Some(degree),
            offset,
        )
        .unwrap()
    }

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
